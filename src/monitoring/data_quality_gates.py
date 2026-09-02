"""
src/monitoring/data_quality_gates.py

PRIORIDAD 2 de Monitoring: Calidad de datos en producción.

Toma un lote (batch) de datos, genera una COPIA contaminada a propósito con
problemas típicos de producción, y corre un conjunto de reglas de validación
que:

    Problema  ->  Detecta  ->  Advierte/Bloquea  ->  Registra

IMPORTANTE:
    - Nunca se modifica el dataset original (data/raw/, data/processed/).
    - La contaminación se hace sobre una copia en memoria.
    - Cada incidente detectado se registra en logs/data_quality_incidents.log

Ejecutar como demo:
    python src/monitoring/data_quality_gates.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "data_quality_incidents.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("data_quality_gates")


# ---------------------------------------------------------------------------
# Paso 1: Contaminar una COPIA del batch (nunca el original)
# ---------------------------------------------------------------------------
def contaminate_batch(df: pd.DataFrame, random_state: int = 7) -> pd.DataFrame:
    """
    Introduce, a propósito, varios problemas típicos de datos "sucios" que
    podrían aparecer en producción. Trabaja SIEMPRE sobre una copia.
    """
    contaminated = df.copy(deep=True)
    rng = np.random.default_rng(random_state)

    # 1. Missing values: se "borran" algunos valores de una columna clave.
    idx_missing = rng.choice(contaminated.index, size=max(1, len(contaminated) // 20), replace=False)
    contaminated.loc[idx_missing, "occupation"] = np.nan

    # 2. Duplicated rows: se repiten algunas filas tal cual.
    duplicated_rows = contaminated.sample(n=max(1, len(contaminated) // 25), random_state=random_state)
    contaminated = pd.concat([contaminated, duplicated_rows], ignore_index=True)

    # 3. Extreme outlier: una edad imposible.
    idx_outlier = rng.choice(contaminated.index, size=1, replace=False)
    contaminated.loc[idx_outlier, "age"] = 999

    # 4. Incorrect datatype: texto donde debería haber un número.
    idx_bad_type = rng.choice(contaminated.index, size=1, replace=False)
    contaminated["age"] = contaminated["age"].astype(object)
    contaminated.loc[idx_bad_type, "age"] = "treinta"

    # 5. Unknown category: un país que no debería existir.
    idx_unknown_cat = rng.choice(contaminated.index, size=1, replace=False)
    contaminated.loc[idx_unknown_cat, "native-country"] = "UNKNOWN_NEW_COUNTRY"

    # 6. Schema modification: aparece una columna que no debería existir.
    contaminated["unexpected_new_field"] = "algo_no_esperado"

    logger.info(
        "Batch contaminado generado (%s filas). Problemas introducidos: "
        "missing values, duplicados, outlier extremo, tipo de dato incorrecto, "
        "categoría desconocida, cambio de esquema.",
        len(contaminated),
    )
    return contaminated


# ---------------------------------------------------------------------------
# Paso 2: Reglas de validación (Data Quality Gates)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

VALID_NATIVE_COUNTRIES = {
    "United-States", "Mexico", "Philippines", "Germany", "Puerto-Rico",
    "Canada", "India", "El-Salvador", "Cuba", "England", "China", "South",
    "Jamaica", "Italy", "Dominican-Republic", "Japan", "Guatemala", "Poland",
    "Vietnam", "Columbia", "Haiti", "Portugal", "Taiwan", "Iran", "Nicaragua",
    "Greece", "Peru", "Ecuador", "France", "Ireland", "Hong", "Thailand",
    "Cambodia", "Trinadad&Tobago", "Laos", "Outlying-US(Guam-USVI-etc)",
    "Yugoslavia", "Scotland", "Honduras", "Hungary", "Holand-Netherlands", "?",
}


def validate_batch(df: pd.DataFrame) -> list[dict]:
    """
    Corre un conjunto de reglas sobre el batch y devuelve una lista de
    incidentes encontrados. Cada incidente incluye:
        - regla que falló
        - severidad (BLOCK = crítico, no se puede procesar el batch;
                     WARN  = se puede procesar pero hay que revisar)
        - detalle del problema
    """
    incidents = []

    # Regla 1: esquema — no deben aparecer columnas inesperadas.
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        incidents.append({
            "rule": "schema_check",
            "severity": "BLOCK",
            "detail": f"Columnas no esperadas en el esquema: {extra_cols}",
        })

    # Regla 2: tipo de dato — age debe poder convertirse a numérico.
    non_numeric_age = pd.to_numeric(df["age"], errors="coerce").isna() & df["age"].notna()
    if non_numeric_age.any():
        incidents.append({
            "rule": "dtype_check_age",
            "severity": "BLOCK",
            "detail": f"{non_numeric_age.sum()} valores de 'age' no son numéricos "
                      f"(ej: {df.loc[non_numeric_age, 'age'].unique()[:3].tolist()})",
        })

    # Regla 3: rangos — age debe estar entre 16 y 100 (ignora los no-numéricos,
    # ya cubiertos por la regla anterior).
    numeric_age = pd.to_numeric(df["age"], errors="coerce")
    out_of_range = numeric_age.dropna().between(16, 100) == False
    if out_of_range.any():
        incidents.append({
            "rule": "range_check_age",
            "severity": "BLOCK",
            "detail": f"{out_of_range.sum()} valores de 'age' fuera de rango (16-100), "
                      f"ej: {numeric_age.dropna()[out_of_range].unique()[:3].tolist()}",
        })

    # Regla 4: missing values — no debería haber nulos en 'occupation'.
    missing_occupation = df["occupation"].isna().sum()
    if missing_occupation > 0:
        incidents.append({
            "rule": "missing_values_occupation",
            "severity": "WARN",
            "detail": f"{missing_occupation} valores faltantes en 'occupation'",
        })

    # Regla 5: duplicados — no debería haber filas 100% repetidas.
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        incidents.append({
            "rule": "duplicated_rows",
            "severity": "WARN",
            "detail": f"{dup_count} filas duplicadas encontradas",
        })

    # Regla 6: categoría desconocida — native-country debe estar en el catálogo conocido.
    unknown_countries = set(df["native-country"].dropna().unique()) - VALID_NATIVE_COUNTRIES
    if unknown_countries:
        incidents.append({
            "rule": "unknown_category_native_country",
            "severity": "WARN",
            "detail": f"Categorías nuevas/desconocidas en 'native-country': {unknown_countries}",
        })

    return incidents


# ---------------------------------------------------------------------------
# Paso 3: Registrar el resultado (Detecta -> Advierte/Bloquea -> Registra)
# ---------------------------------------------------------------------------
def process_batch_with_gates(df: pd.DataFrame) -> str:
    """
    Corre las validaciones y decide si el batch se BLOQUEA, pasa con
    ADVERTENCIA, o se ACEPTA sin problemas. Registra todo en el log.
    Devuelve el estado final como string.
    """
    incidents = validate_batch(df)

    if not incidents:
        logger.info("✅ Batch válido: ninguna regla de calidad falló. Se acepta sin problemas.")
        return "ACCEPTED"

    has_block = any(i["severity"] == "BLOCK" for i in incidents)

    logger.warning("Se detectaron %s incidente(s) de calidad en el batch:", len(incidents))
    for incident in incidents:
        icon = "🔴 BLOCK" if incident["severity"] == "BLOCK" else "🟡 WARN"
        logger.warning("  [%s] %s -> %s", icon, incident["rule"], incident["detail"])

    if has_block:
        logger.error("🔴 Batch BLOQUEADO: contiene al menos un problema crítico. No se procesa.")
        return "BLOCKED"
    else:
        logger.warning("🟡 Batch ACEPTADO CON ADVERTENCIAS: se procesa, pero requiere revisión.")
        return "ACCEPTED_WITH_WARNINGS"


def run_data_quality_demo():
    from src.monitoring.drift_detection import load_raw_data, build_reference_and_batches

    df = load_raw_data()
    batches = build_reference_and_batches(df)

    # Usamos el lote 2 como base para la contaminación (podría ser cualquiera).
    clean_batch = batches["lote_2_moderado"]

    print("=" * 70)
    print("DATA QUALITY GATES - Simulación de contaminación sobre una copia")
    print("=" * 70)
    print(f"\nBatch limpio original: {clean_batch.shape[0]} filas "
          f"(este NO se modifica en ningún momento)")

    contaminated_batch = contaminate_batch(clean_batch)
    print(f"Batch contaminado (copia): {contaminated_batch.shape[0]} filas\n")

    final_status = process_batch_with_gates(contaminated_batch)
    print(f"\n>>> Estado final del batch: {final_status}")
    print(f">>> Log completo disponible en: {LOG_DIR / 'data_quality_incidents.log'}")

    # Confirmación explícita de que el dataset original sigue intacto.
    original_check = load_raw_data()
    assert original_check.shape == df.shape, "¡El dataset original fue modificado!"
    print(">>> Verificado: el dataset original (data/raw/) permanece intacto.")

    return final_status


if __name__ == "__main__":
    run_data_quality_demo()
