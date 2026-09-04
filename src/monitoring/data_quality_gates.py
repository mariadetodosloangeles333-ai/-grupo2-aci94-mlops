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
from pathlib import Path

import numpy as np
import pandas as pd

# Asegura que el directorio raíz del proyecto esté en el path de Python,
# para que "from src.monitoring.drift_detection import ..." funcione sin
# importar desde dónde se ejecute este script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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

    # 5. Categoría desconocida en una variable utilizada por el modelo.
    idx_unknown_cat = rng.choice(
        contaminated.index,
        size=1,
        replace=False,
    )
    contaminated.loc[
        idx_unknown_cat,
        "occupation",
    ] = "UNKNOWN_NEW_OCCUPATION"

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
    "age",
    "education-num",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
]

VALID_OCCUPATIONS = {
    "Adm-clerical",
    "Armed-Forces",
    "Craft-repair",
    "Exec-managerial",
    "Farming-fishing",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Other-service",
    "Priv-house-serv",
    "Prof-specialty",
    "Protective-serv",
    "Sales",
    "Tech-support",
    "Transport-moving",
    "Unknown",
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

    # Regla 1: esquema — deben estar todas las columnas productivas
    # y no deben aparecer columnas adicionales.
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_cols:
        incidents.append(
            {
                "rule": "missing_required_columns",
                "severity": "BLOCK",
                "detail": (
                    "Faltan columnas requeridas en el esquema productivo: "
                    f"{sorted(missing_cols)}"
                ),
            }
        )

    if extra_cols:
        incidents.append(
            {
                "rule": "unexpected_columns",
                "severity": "BLOCK",
                "detail": (
                    "Se encontraron columnas no esperadas en el esquema "
                    f"productivo: {sorted(extra_cols)}"
                ),
            }
        )

    # Regla 2: tipo de dato — age debe poder convertirse a numérico.
    # La comprobación se realiza únicamente si la columna está presente;
    # su ausencia ya fue registrada como una violación de esquema.
    if "age" in df.columns:
        numeric_age = pd.to_numeric(
            df["age"],
            errors="coerce",
        )

        non_numeric_age = (
            numeric_age.isna()
            & df["age"].notna()
        )

        if non_numeric_age.any():
            invalid_examples = (
                df.loc[
                    non_numeric_age,
                    "age",
                ]
                .unique()[:3]
                .tolist()
            )

            incidents.append(
                {
                    "rule": "dtype_check_age",
                    "severity": "BLOCK",
                    "detail": (
                        f"{int(non_numeric_age.sum())} valores de 'age' "
                        "no son numéricos "
                        f"(ejemplos: {invalid_examples})"
                    ),
                }
            )

        # Regla 3: age debe coincidir con el rango aceptado por FastAPI.
        valid_numeric_age = numeric_age.dropna()
        out_of_range = ~valid_numeric_age.between(
            17,
            90,
        )

        if out_of_range.any():
            outlier_examples = (
                valid_numeric_age.loc[
                    out_of_range
                ]
                .unique()[:3]
                .tolist()
            )

            incidents.append(
                {
                    "rule": "range_check_age",
                    "severity": "BLOCK",
                    "detail": (
                        f"{int(out_of_range.sum())} valores de 'age' "
                        "fuera de rango (17-90) "
                        f"(ejemplos: {outlier_examples})"
                    ),
                }
            )

    # Regla 4: valores faltantes en una variable productiva.
    # La ausencia completa de occupation ya se controla en el esquema.
    if "occupation" in df.columns:
        missing_occupation = int(
            df["occupation"].isna().sum()
        )

        if missing_occupation > 0:
            incidents.append(
                {
                    "rule": "missing_values_occupation",
                    "severity": "WARN",
                    "detail": (
                        f"{missing_occupation} valores faltantes "
                        "en 'occupation'"
                    ),
                }
            )

    # Regla 5: duplicados — no debería haber filas 100% repetidas.
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        incidents.append({
            "rule": "duplicated_rows",
            "severity": "WARN",
            "detail": f"{dup_count} filas duplicadas encontradas",
        })

    # Regla 6: categoría nueva en una variable productiva.
    if "occupation" in df.columns:
        observed_occupations = set(
            df["occupation"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unknown_occupations = (
            observed_occupations
            - VALID_OCCUPATIONS
        )

        if unknown_occupations:
            incidents.append(
                {
                    "rule": "unknown_category_occupation",
                    "severity": "WARN",
                    "detail": (
                        "Categorías nuevas o desconocidas en "
                        f"'occupation': {sorted(unknown_occupations)}"
                    ),
                }
            )

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
    raw_batch = batches[
        "lote_2_moderado"
    ]

    # Monitoring utiliza únicamente el contrato de entrada de Production v2.
    clean_batch = raw_batch.loc[
        :,
        EXPECTED_COLUMNS,
    ].copy()

    # Simular la estructura que llegaría al servicio después de normalizar
    # las categorías faltantes conocidas.
    categorical_columns = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
    ]

    for column in categorical_columns:
        clean_batch[column] = (
            clean_batch[column]
            .replace("?", np.nan)
            .fillna("Unknown")
        )

    # El batch base debe iniciar sin duplicados para que los duplicados
    # detectados correspondan únicamente a la contaminación simulada.
    clean_batch = (
        clean_batch
        .drop_duplicates()
        .reset_index(drop=True)
    )

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
    pd.testing.assert_frame_equal(original_check, df)
    print(">>> Verificado: el dataset original (data/raw/) permanece intacto.")

    return final_status


if __name__ == "__main__":
    run_data_quality_demo()
