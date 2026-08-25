# Pipeline reproducible de limpieza

# Importar librerías
import logging
import sys
from pathlib import Path
import pandas as pd

# Permitir ejecutar clean.py directamente o importarlo desde las pruebas
try:
    from src.cleaning.data_quality_gates import (
        ALLOWED_MISSING_COUNTS,
        CATEGORICAL_COLUMNS,
        EXPECTED_COLUMNS,
        run_clean_gates,
        run_reference_gates,
    )
except ModuleNotFoundError:
    from data_quality_gates import (
        ALLOWED_MISSING_COUNTS,
        CATEGORICAL_COLUMNS,
        EXPECTED_COLUMNS,
        run_clean_gates,
        run_reference_gates,
    )

# Ubicar la carpeta principal del repositorio
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Definir las rutas de entrada, salida y logs
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult_raw.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "adult_clean.csv"
LOG_DIR = PROJECT_ROOT / "logs"

# Crear las carpetas necesarias si todavía no existen
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configurar el registro de eventos en la terminal y en un archivo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / "cleaning.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger("cleaning")

# Función para cargar los datos crudos con el script de ingesta
def load_raw_data() -> pd.DataFrame:

    # Comprobar que el archivo raw exista
    if not RAW_DATA_PATH.exists():
        message = (
            f"No se encontró el archivo raw en: {RAW_DATA_PATH}. "
            "Ejecute primero: python src/ingestion/ingest.py"
        )
        logger.error(message)
        raise FileNotFoundError(message)

    # Cargar el archivo sin modificarlo
    df = pd.read_csv(RAW_DATA_PATH)

    # Registrar la cantidad de filas y columnas recibidas
    logger.info(
        "Dataset raw cargado: %s filas, %s columnas.",
        df.shape[0],
        df.shape[1],
    )

    return df

def log_gate_report(report: dict, stage: str) -> None:
    """
    Registra los resultados individuales y el resumen de una validación.
    """

    for result in report["results"]:
        gate_name = result["gate"]
        status = result["status"]
        message = result["message"]

        if status == "FAIL":
            logger.error(
                "[%s][%s] %s: %s",
                stage,
                gate_name,
                status,
                message,
            )

        elif status == "WARNING":
            logger.warning(
                "[%s][%s] %s: %s",
                stage,
                gate_name,
                status,
                message,
            )

        else:
            logger.info(
                "[%s][%s] %s: %s",
                stage,
                gate_name,
                status,
                message,
            )

    logger.info(
        "[%s] Resumen: PASS=%s, WARNING=%s, FAIL=%s.",
        stage,
        report["summary"]["PASS"],
        report["summary"]["WARNING"],
        report["summary"]["FAIL"],
    )


def validate_reference_data(df: pd.DataFrame) -> dict:
    """
    Ejecuta y registra los Data Quality Gates antes de la limpieza.
    """

    logger.info("Iniciando Data Quality Gates de referencia.")

    report = run_reference_gates(df)
    log_gate_report(report, stage="RAW")

    if report["overall_status"] == "FAIL":
        message = (
            "Los datos raw no superaron los Data Quality Gates. "
            "La limpieza fue bloqueada."
        )
        logger.error(message)
        raise ValueError(message)

    logger.info(
        "Validación raw finalizada con estado general: %s.",
        report["overall_status"],
    )

    return report


def validate_clean_data(df: pd.DataFrame) -> dict:
    """
    Ejecuta y registra los Data Quality Gates después de la limpieza.
    """

    logger.info("Iniciando Data Quality Gates posteriores a la limpieza.")

    report = run_clean_gates(df)
    log_gate_report(report, stage="CLEAN")

    if report["overall_status"] == "FAIL":
        message = (
            "Los datos limpios no superaron los Data Quality Gates. "
            "El archivo procesado no será guardado."
        )
        logger.error(message)
        raise ValueError(message)

    logger.info(
        "Validación de salida finalizada con estado general: %s.",
        report["overall_status"],
    )

    return report

# Se aplica la limpieza definida en el diagnóstico
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Iniciando limpieza de datos.")

    # Crear una copia para no modificar el DataFrame raw
    df_clean = df.copy()

    # Conservar únicamente las columnas aprobadas, respetando su orden original
    approved_columns = [
        column
        for column in df_clean.columns
        if column in EXPECTED_COLUMNS
    ]
    df_clean = df_clean.loc[:, approved_columns].copy()

    # Limpiar espacios y convertir "?" en valores faltantes reconocidos
    for column in CATEGORICAL_COLUMNS:
        normalized_values = (
            df_clean[column]
            .astype("string")
            .str.strip()
        )

        df_clean[column] = normalized_values.mask(
            normalized_values.eq("?"),
            pd.NA,
        )

    # Crear la categoría Unknown únicamente en las variables aprobadas
    for column in ALLOWED_MISSING_COUNTS:
        df_clean[column] = df_clean[column].fillna("Unknown")

    # Normalizar las cuatro etiquetas de income a las dos clases finales
    df_clean["income"] = df_clean["income"].replace(
        {
            "<=50K.": "<=50K",
            ">50K.": ">50K",
        }
    )

    # Contar los duplicados después de normalizar todas las categorías
    duplicate_count = int(df_clean.duplicated().sum())

    # Eliminar duplicados y reconstruir el índice
    df_clean = (
        df_clean
        .drop_duplicates()
        .reset_index(drop=True)
    )

    logger.info(
        "Limpieza aplicada: %s duplicados eliminados.",
        duplicate_count,
    )

    logger.info(
        "Dataset limpio: %s filas, %s columnas.",
        df_clean.shape[0],
        df_clean.shape[1],
    )

    return df_clean

# Función para guardar el dataset limpio después de superar los gates de salida
def save_clean_data(df: pd.DataFrame) -> Path:

    # Guardar sin incluir el índice de Pandas
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    logger.info(
        "Dataset procesado guardado en: %s",
        PROCESSED_DATA_PATH,
    )

    return PROCESSED_DATA_PATH

# Función para ejecutar el pipeline de validación y limpieza
def main() -> None:

    logger.info("=== Iniciando pipeline de limpieza ===")

    try:
        # Cargar los datos producidos por la ingesta
        raw_df = load_raw_data()

        # Validar los datos antes de aplicar transformaciones
        validate_reference_data(raw_df)

        # Aplicar las decisiones de limpieza
        clean_df = clean_data(raw_df)

        # Comprobar que la limpieza produjo el resultado esperado
        validate_clean_data(clean_df)

        # Guardar únicamente si no existen fallos
        output_path = save_clean_data(clean_df)

        logger.info(
            "=== Pipeline finalizado con éxito: %s ===",
            output_path,
        )

    except Exception:
        # Registrar cualquier error antes de detener el proceso
        logger.exception("=== El pipeline de limpieza falló ===")
        raise


if __name__ == "__main__":
    main()

# Flujo de este trabajo

# 1. Se crearon los Data Quality Gates.
# 2. Se verifican los gates de entrada y se obtiene un resultado.
# 3. Según el resultado, se aplica la limpieza definida en el diagnóstico.
# 4. Se verifican los gates de salida para comprobar que la limpieza funcionó.
# 5. Se guarda el dataset limpio.

# Si ocurre un FAIL en cualquiera de los gates, se detiene el pipeline
# y no se guarda el dataset.