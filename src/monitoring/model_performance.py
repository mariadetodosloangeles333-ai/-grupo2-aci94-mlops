"""
Monitoreo del desempeño del modelo sobre batches etiquetados.

Este módulo reutiliza las mismas métricas empleadas durante el entrenamiento.
Solo puede evaluar desempeño cuando el ground truth del batch ya está
disponible; no utiliza el conjunto de test ni reentrena el modelo.

Ejecutar como demo:
    python src/monitoring/model_performance.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.monitoring.retraining_decision import (  # noqa: E402
    PERFORMANCE_DROP_THRESHOLD,
    PRODUCTION_G_MEAN,
)
from src.training.train import (  # noqa: E402
    calculate_classification_metrics,
)


METRICS_TO_MONITOR = (
    "precision",
    "recall",
    "specificity",
    "f1",
    "g_mean",
    "roc_auc",
)


def evaluate_labeled_batch(
    batch_id: str,
    y_true,
    y_pred,
    y_probability,
) -> dict:
    """
    Calcula las métricas de un batch cuando sus etiquetas están disponibles.

    Parámetros:
        batch_id: identificador único del lote o periodo evaluado.
        y_true: etiquetas reales conocidas posteriormente.
        y_pred: clases predichas por el modelo productivo.
        y_probability: probabilidad asignada a la clase positiva >50K.

    Devuelve un registro temporal con las métricas y la comparación del
    G-Mean actual contra el baseline aprobado de Production v1.
    """
    if not batch_id or not batch_id.strip():
        raise ValueError("batch_id no puede estar vacío.")

    sample_count = len(y_true)

    if sample_count == 0:
        raise ValueError("El batch etiquetado no puede estar vacío.")

    if len(y_pred) != sample_count or len(y_probability) != sample_count:
        raise ValueError(
            "y_true, y_pred y y_probability deben tener la misma longitud."
        )

    y_true_series = pd.Series(y_true)

    if y_true_series.isna().any():
        raise ValueError(
            "y_true no puede contener etiquetas faltantes."
        )

    if y_true_series.nunique() < 2:
        raise ValueError(
            "El batch debe contener ambas clases para calcular "
            "G-Mean y ROC AUC de forma interpretable."
        )

    all_metrics = calculate_classification_metrics(
        y_true=y_true_series,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    monitored_metrics = {
        metric_name: float(all_metrics[metric_name])
        for metric_name in METRICS_TO_MONITOR
    }

    current_g_mean = monitored_metrics["g_mean"]
    g_mean_drop = PRODUCTION_G_MEAN - current_g_mean

    return {
        "batch_id": batch_id,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": sample_count,
        "ground_truth_available": True,
        "metrics": monitored_metrics,
        "baseline_g_mean": PRODUCTION_G_MEAN,
        "g_mean_drop": float(g_mean_drop),
        "performance_drop_threshold": PERFORMANCE_DROP_THRESHOLD,
        "performance_degraded": (
            g_mean_drop >= PERFORMANCE_DROP_THRESHOLD
        ),
    }


def run_model_performance_demo():
    """
    Evalúa dos batches ilustrativos con ground truth simulado.

    Estos resultados no representan tráfico real ni sustituyen la evaluación
    final del modelo; demuestran cómo se monitorearían métricas en el tiempo
    cuando las etiquetas productivas estuvieran disponibles.
    """
    y_true = pd.Series(
        [
            "<=50K",
            "<=50K",
            "<=50K",
            "<=50K",
            "<=50K",
            ">50K",
            ">50K",
            ">50K",
            ">50K",
            ">50K",
        ]
    )

    scenarios = [
        {
            "batch_id": "production_batch_001_stable",
            "y_pred": [
                "<=50K",
                "<=50K",
                "<=50K",
                "<=50K",
                ">50K",
                "<=50K",
                ">50K",
                ">50K",
                ">50K",
                ">50K",
            ],
            "y_probability": [
                0.10,
                0.20,
                0.30,
                0.40,
                0.60,
                0.45,
                0.65,
                0.75,
                0.85,
                0.90,
            ],
        },
        {
            "batch_id": "production_batch_002_degraded",
            "y_pred": [
                "<=50K",
                "<=50K",
                "<=50K",
                ">50K",
                ">50K",
                "<=50K",
                "<=50K",
                "<=50K",
                ">50K",
                ">50K",
            ],
            "y_probability": [
                0.20,
                0.30,
                0.40,
                0.60,
                0.70,
                0.30,
                0.35,
                0.45,
                0.65,
                0.75,
            ],
        },
    ]

    print("=" * 70)
    print("MODEL PERFORMANCE MONITORING - Batches etiquetados simulados")
    print("=" * 70)

    for scenario in scenarios:
        result = evaluate_labeled_batch(
            batch_id=scenario["batch_id"],
            y_true=y_true,
            y_pred=scenario["y_pred"],
            y_probability=scenario["y_probability"],
        )

        print(f"\nBatch: {result['batch_id']}")
        print(f"Muestras etiquetadas: {result['sample_count']}")

        for metric_name, metric_value in result["metrics"].items():
            print(f"  - {metric_name:12s}: {metric_value:.4f}")

        print(f"Caída de G-Mean: {result['g_mean_drop']:.4f}")
        print(f"Desempeño deteriorado: {result['performance_degraded']}")


if __name__ == "__main__":
    run_model_performance_demo()