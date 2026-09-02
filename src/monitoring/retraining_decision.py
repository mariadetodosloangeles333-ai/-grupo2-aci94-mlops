"""
src/monitoring/retraining_decision.py

PRIORIDAD 3 de Monitoring: Decisión de reentrenamiento.

Idea central que pide la rúbrica y que este módulo demuestra con código:

    Drift ≠ Model Degradation.

Que la distribución de los datos haya cambiado (drift) NO significa
automáticamente que el modelo esté funcionando peor. Por eso la decisión
de reentrenar combina DOS señales, no una sola:

    1. ¿Hay drift? (PSI, de drift_detection.py)
    2. ¿El desempeño del modelo se deterioró? (métrica de negocio, ej. F1/AUC)

La combinación de ambas produce una de tres recomendaciones:

    MANTENER                      -> sin drift relevante y desempeño estable
    REVISAR                       -> hay drift pero el desempeño sigue bien
                                      (monitorear de cerca, no actuar aún)
    CONSIDERAR REENTRENAMIENTO    -> hay drift relevante Y el desempeño
                                      se deterioró de forma significativa

Ejecutar como demo:
    python src/monitoring/retraining_decision.py
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Umbrales (documentados y ajustables, no son ley universal)
# ---------------------------------------------------------------------------
# Se usa el umbral de WARNING (no el de ALERT) como disparador de "hay drift"
# para esta decisión: incluso un cambio moderado de distribución merece
# cruzarse con el desempeño, no solo los cambios extremos.
PSI_DRIFT_THRESHOLD = 0.10          # mismo umbral WARNING usado en drift_detection.py
PERFORMANCE_DROP_THRESHOLD = 0.05   # una caída de 5 puntos porcentuales en la
                                     # métrica principal (ej. F1) se considera
                                     # deterioro significativo del modelo.


@dataclass
class RetrainingDecision:
    decision: str
    reason: str
    max_psi: float
    baseline_metric: float
    current_metric: float
    performance_drop: float


def decide_retraining(
    max_psi: float,
    baseline_metric: float,
    current_metric: float,
    psi_threshold: float = PSI_DRIFT_THRESHOLD,
    performance_drop_threshold: float = PERFORMANCE_DROP_THRESHOLD,
) -> RetrainingDecision:
    """
    Combina drift (PSI) y desempeño del modelo para decidir qué hacer.

    Parámetros:
        max_psi: el PSI más alto detectado entre las columnas monitoreadas
                 (viene de drift_detection.evaluate_drift_for_batch).
        baseline_metric: métrica de desempeño del modelo cuando se entrenó
                 (ej. F1-score en el set de validación original).
        current_metric: métrica de desempeño del modelo en el batch de
                 producción actual (requiere tener ground truth, aunque sea
                 de una muestra etiquetada posteriormente).

    Lógica:
        - Sin drift relevante (PSI < umbral):
              el desempeño manda. Si igual cayó mucho, hay que revisar
              (podría ser un problema de otra naturaleza, no de distribución).
        - Con drift relevante (PSI >= umbral):
              si el desempeño se mantiene, es una señal de que el modelo
              sigue generalizando bien a pesar del cambio -> solo revisar.
              si el desempeño también cayó, ahí sí se junta drift real +
              impacto medible -> considerar reentrenamiento.
    """
    performance_drop = baseline_metric - current_metric
    has_drift = max_psi >= psi_threshold
    has_performance_degradation = performance_drop >= performance_drop_threshold

    if not has_drift and not has_performance_degradation:
        decision = "MANTENER"
        reason = (
            f"No hay drift relevante (PSI={max_psi:.3f} < {psi_threshold}) y el "
            f"desempeño se mantiene estable (caída de {performance_drop:.3f}, "
            f"por debajo del umbral de {performance_drop_threshold}). "
            "El modelo sigue siendo confiable."
        )

    elif has_drift and not has_performance_degradation:
        decision = "REVISAR"
        reason = (
            f"Se detectó drift (PSI={max_psi:.3f} >= {psi_threshold}) pero el "
            f"desempeño del modelo se mantiene estable (caída de {performance_drop:.3f}). "
            "Esto confirma que Drift != Model Degradation: la distribución de entrada "
            "cambió, pero el modelo sigue prediciendo bien sobre esos datos. "
            "Se recomienda vigilar de cerca en los próximos batches antes de actuar."
        )

    elif not has_drift and has_performance_degradation:
        decision = "REVISAR"
        reason = (
            f"No se detectó drift significativo (PSI={max_psi:.3f} < {psi_threshold}) "
            f"pero el desempeño cayó {performance_drop:.3f}, por encima del umbral. "
            "Esto sugiere que la causa NO es un cambio de distribución de entrada "
            "(podría ser un bug, un cambio en el pipeline de features, o un "
            "problema de calidad de datos) — hay que investigar antes de reentrenar, "
            "porque reentrenar no resolvería la causa raíz si no es drift."
        )

    else:  # has_drift and has_performance_degradation
        decision = "CONSIDERAR REENTRENAMIENTO"
        reason = (
            f"Hay drift relevante (PSI={max_psi:.3f} >= {psi_threshold}) Y el "
            f"desempeño se deterioró de forma significativa (caída de "
            f"{performance_drop:.3f} >= {performance_drop_threshold}). "
            "Ambas señales juntas indican que el cambio en los datos SÍ está "
            "afectando la capacidad predictiva del modelo. Este es el escenario "
            "en el que reentrenar tiene sentido."
        )

    return RetrainingDecision(
        decision=decision,
        reason=reason,
        max_psi=max_psi,
        baseline_metric=baseline_metric,
        current_metric=current_metric,
        performance_drop=performance_drop,
    )


def run_retraining_decision_demo():
    """
    Corre la lógica sobre 3 escenarios de ejemplo, reflejando los 3 lotes
    de drift_detection.py, para mostrar cómo la misma señal de drift puede
    llevar a decisiones distintas según el desempeño observado.
    """
    scenarios = [
        {
            "name": "Lote 1 (normal) — sin drift, desempeño estable",
            "max_psi": 0.02, "baseline_metric": 0.79, "current_metric": 0.78,
        },
        {
            "name": "Lote 2 (moderado) — drift, pero el modelo sigue funcionando bien",
            "max_psi": 0.14, "baseline_metric": 0.79, "current_metric": 0.77,
        },
        {
            "name": "Lote 3 (fuerte) — drift alto Y el desempeño se derrumbó",
            "max_psi": 0.45, "baseline_metric": 0.79, "current_metric": 0.68,
        },
    ]

    print("=" * 70)
    print("DECISIÓN DE REENTRENAMIENTO — Drift ≠ Model Degradation")
    print("=" * 70)

    for scenario in scenarios:
        result = decide_retraining(
            max_psi=scenario["max_psi"],
            baseline_metric=scenario["baseline_metric"],
            current_metric=scenario["current_metric"],
        )
        print(f"\n{scenario['name']}")
        print(f"  -> Decisión: {result.decision}")
        print(f"  -> Justificación: {result.reason}")


if __name__ == "__main__":
    run_retraining_decision_demo()
