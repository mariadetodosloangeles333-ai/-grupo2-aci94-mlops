"""
src/monitoring/drift_detection.py

PRIORIDAD 1 de Monitoring: Drift.

Construye:
    - Un REFERENCE / BASELINE a partir del dataset histórico (adult_raw.csv)
    - 3 lotes de "producción" simulados:
        Lote 1 (normal)   -> se espera PSI bajo   -> 🟢 OK
        Lote 2 (moderado) -> se espera PSI medio   -> 🟡 WARNING
        Lote 3 (fuerte)   -> se espera PSI alto    -> 🔴 ALERT

Mide el cambio de distribución con PSI (Population Stability Index),
la métrica más estándar y fácil de justificar para este tipo de proyecto.

IMPORTANTE: este script NUNCA modifica data/raw/ ni data/processed/.
Todos los "lotes" se generan en memoria (o se guardan aparte en
data/monitoring/ si se quiere dejar evidencia), a partir de copias.

Ejecutar como demo:
    python src/monitoring/drift_detection.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = "data/raw/adult_raw.csv"

# ---------------------------------------------------------------------------
# Umbrales de PSI (est\u00e1ndar de la industria, ampliamente citado en la
# literatura de model monitoring; no son ley universal, pero s\u00ed un punto
# de partida razonable y f\u00e1cil de justificar):
#
#   PSI < 0.10            -> sin cambio significativo               -> OK
#   0.10 <= PSI < 0.25     -> cambio moderado, vigilar de cerca       -> WARNING
#   PSI >= 0.25            -> cambio fuerte, la distribución cambió   -> ALERT
# ---------------------------------------------------------------------------
PSI_THRESHOLD_WARNING = 0.10
PSI_THRESHOLD_ALERT = 0.25

# Columnas sobre las que vamos a monitorear drift (mezcla de numérica y
# categórica, para mostrar que la técnica aplica a ambos tipos).
NUMERIC_COLUMNS_TO_MONITOR = ["age", "hours-per-week"]
CATEGORICAL_COLUMNS_TO_MONITOR = ["education"]


# ---------------------------------------------------------------------------
# Cálculo de PSI
# ---------------------------------------------------------------------------
def calculate_psi_numeric(reference: pd.Series, production: pd.Series, buckets: int = 10) -> float:
    """
    PSI para una variable numérica, usando los mismos cortes (quantiles)
    del reference para dividir ambas distribuciones en buckets.
    """
    reference = reference.dropna()
    production = production.dropna()

    # Cortes basados en cuantiles del REFERENCE (no del batch de producción,
    # para que la comparación sea justa).
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(reference.quantile(quantiles).values)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = pd.cut(reference, bins=breakpoints).value_counts(normalize=True).sort_index()
    prod_counts = pd.cut(production, bins=breakpoints).value_counts(normalize=True).sort_index()

    return _psi_from_proportions(ref_counts, prod_counts)


def calculate_psi_categorical(reference: pd.Series, production: pd.Series) -> float:
    """PSI para una variable categórica, usando las categorías como buckets."""
    ref_counts = reference.value_counts(normalize=True)
    prod_counts = production.value_counts(normalize=True)

    all_categories = set(ref_counts.index) | set(prod_counts.index)
    ref_counts = ref_counts.reindex(all_categories, fill_value=0)
    prod_counts = prod_counts.reindex(all_categories, fill_value=0)

    return _psi_from_proportions(ref_counts, prod_counts)


def _psi_from_proportions(ref_props: pd.Series, prod_props: pd.Series, epsilon: float = 1e-4) -> float:
    """
    Fórmula estándar de PSI:
        PSI = sum( (prod_% - ref_%) * ln(prod_% / ref_%) )
    Se agrega un epsilon pequeño para evitar log(0) o división entre 0
    en buckets vacíos.
    """
    ref = ref_props.clip(lower=epsilon)
    prod = prod_props.reindex(ref.index).fillna(0).clip(lower=epsilon)
    psi_values = (prod - ref) * np.log(prod / ref)
    return float(psi_values.sum())


def classify_psi(psi_value: float) -> str:
    """Traduce un valor de PSI al semáforo OK / WARNING / ALERT."""
    if psi_value >= PSI_THRESHOLD_ALERT:
        return "ALERT"
    elif psi_value >= PSI_THRESHOLD_WARNING:
        return "WARNING"
    return "OK"


# ---------------------------------------------------------------------------
# Construcción del reference y los 3 lotes simulados
# ---------------------------------------------------------------------------
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


def build_reference_and_batches(df: pd.DataFrame, random_state: int = 42):
    """
    Divide el dataset histórico en:
        - reference: 40% de los datos, tomado como "lo que el modelo conoce"
        - lote_1_normal:   otro 20%, SIN alterar -> distribución muy parecida
        - lote_2_moderado: otro 20%, con una alteración moderada
        - lote_3_fuerte:   otro 20%, con una alteración fuerte

    Las alteraciones son completamente sintéticas y se hacen sobre COPIAS
    en memoria; el dataframe original (df) nunca se modifica.
    """
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df_shuffled)

    reference = df_shuffled.iloc[: int(n * 0.4)].copy()
    lote_1 = df_shuffled.iloc[int(n * 0.4): int(n * 0.6)].copy()
    lote_2 = df_shuffled.iloc[int(n * 0.6): int(n * 0.8)].copy()
    lote_3 = df_shuffled.iloc[int(n * 0.8):].copy()

    # --- Lote 1: normal, no se toca. Debería dar PSI bajo (OK). ---

    # --- Lote 2: cambio MODERADO ---
    # Se envejece la población un poco y se reduce ligeramente hours-per-week,
    # simulando un cambio demográfico gradual (ej. una campaña que atrajo
    # usuarios algo mayores, o temporada con jornadas más cortas).
    lote_2["age"] = (lote_2["age"] + 3).clip(upper=90)
    lote_2["hours-per-week"] = (lote_2["hours-per-week"] * 0.95).clip(lower=1, upper=99).round()
    # Se sobre-representa un poco la categoría "Bachelors" en education
    mask = lote_2.sample(frac=0.10, random_state=random_state).index
    lote_2.loc[mask, "education"] = "Bachelors"

    # --- Lote 3: cambio FUERTE ---
    # Simula un cambio grande en la población de entrada (ej. la fuente de
    # datos cambió de canal/segmento por completo).
    lote_3["age"] = (lote_3["age"] + 20).clip(upper=90)
    lote_3["hours-per-week"] = (lote_3["hours-per-week"] * 1.6).clip(lower=1, upper=99).round()
    mask = lote_3.sample(frac=0.7, random_state=random_state).index
    lote_3.loc[mask, "education"] = "Doctorate"

    return {
        "reference": reference,
        "lote_1_normal": lote_1,
        "lote_2_moderado": lote_2,
        "lote_3_fuerte": lote_3,
    }


def evaluate_drift_for_batch(reference: pd.DataFrame, batch: pd.DataFrame) -> dict:
    """
    Calcula PSI para cada columna monitoreada y devuelve:
        - psi por columna
        - psi máximo (el que manda para la clasificación general del batch)
        - clasificación final (OK / WARNING / ALERT)
    """
    psi_per_column = {}

    for col in NUMERIC_COLUMNS_TO_MONITOR:
        psi_per_column[col] = calculate_psi_numeric(reference[col], batch[col])

    for col in CATEGORICAL_COLUMNS_TO_MONITOR:
        psi_per_column[col] = calculate_psi_categorical(reference[col], batch[col])

    max_psi = max(psi_per_column.values())
    return {
        "psi_per_column": psi_per_column,
        "max_psi": max_psi,
        "status": classify_psi(max_psi),
    }


def run_drift_demo():
    df = load_raw_data()
    batches = build_reference_and_batches(df)
    reference = batches["reference"]

    print("=" * 70)
    print("DRIFT DETECTION - Reference vs 3 lotes de producción simulados")
    print(f"Umbrales -> OK: PSI < {PSI_THRESHOLD_WARNING} | "
          f"WARNING: {PSI_THRESHOLD_WARNING} <= PSI < {PSI_THRESHOLD_ALERT} | "
          f"ALERT: PSI >= {PSI_THRESHOLD_ALERT}")
    print("=" * 70)

    results = {}
    for name in ["lote_1_normal", "lote_2_moderado", "lote_3_fuerte"]:
        result = evaluate_drift_for_batch(reference, batches[name])
        results[name] = result

        emoji = {"OK": "🟢", "WARNING": "🟡", "ALERT": "🔴"}[result["status"]]
        print(f"\n{name} -> {emoji} {result['status']}  (PSI máximo = {result['max_psi']:.4f})")
        for col, psi in result["psi_per_column"].items():
            print(f"    - {col:20s} PSI = {psi:.4f}")

    return results


if __name__ == "__main__":
    run_drift_demo()
