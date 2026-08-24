"""
src/ingestion/ingest.py

Script de ingesta reproducible para el proyecto ACI94 (Grupo 2).

Descarga el Adult / Census Income Dataset directamente desde el UCI Machine
Learning Repository (no depende de un CSV guardado a mano en ninguna compu),
le asigna los nombres de columna correctos, y lo guarda en data/raw/.

Uso:
    python src/ingestion/ingest.py

Requisitos:
    pip install ucimlrepo pandas

Fuente oficial:
    Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning
    Repository. https://doi.org/10.24432/C5XW20
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

UCI_DATASET_ID = 2  # id oficial del Adult dataset en el UCI ML Repository
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ingest")


def download_adult_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Descarga el Adult dataset desde el UCI ML Repository usando el paquete
    oficial `ucimlrepo`. Devuelve (features, targets, metadata).
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        logger.error(
            "Falta el paquete 'ucimlrepo'. Instalalo con: pip install ucimlrepo"
        )
        raise exc

    logger.info("Descargando dataset Adult (id=%s) desde UCI ML Repository...", UCI_DATASET_ID)
    adult = fetch_ucirepo(id=UCI_DATASET_ID)

    features = adult.data.features
    targets = adult.data.targets
    metadata = {
        "name": adult.metadata.get("name"),
        "uci_id": adult.metadata.get("uci_id"),
        "num_instances": adult.metadata.get("num_instances"),
        "num_features": adult.metadata.get("num_features"),
        "doi": adult.metadata.get("doi"),
    }

    logger.info(
        "Descarga completa: %s filas, %s columnas de features",
        features.shape[0],
        features.shape[1],
    )
    return features, targets, metadata


def build_raw_dataframe(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Une features + target en un único DataFrame crudo, tal cual llega de la fuente."""
    df = features.copy()
    target_col = targets.columns[0]
    df["income"] = targets[target_col]
    return df


def validate_minimum_shape(df: pd.DataFrame) -> None:
    """
    Validación mínima de ingesta (no reemplaza el diagnóstico de calidad de
    la Etapa 3, solo confirma que la descarga trajo lo esperado).
    """
    expected_min_rows = 48000  # el dataset completo trae 48,842 filas
    expected_cols = 15  # 14 features + income

    assert df.shape[0] >= expected_min_rows, (
        f"Se esperaban al menos {expected_min_rows} filas, llegaron {df.shape[0]}"
    )
    assert df.shape[1] == expected_cols, (
        f"Se esperaban {expected_cols} columnas, llegaron {df.shape[1]}"
    )
    logger.info("Validación mínima de ingesta: OK (%s filas, %s columnas)", df.shape[0], df.shape[1])


def save_raw(df: pd.DataFrame, metadata: dict) -> Path:
    """Guarda el dataset crudo en data/raw/ con timestamp de ingesta en el log."""
    output_path = RAW_DATA_DIR / "adult_raw.csv"
    df.to_csv(output_path, index=False)

    logger.info("Dataset guardado en: %s", output_path)
    logger.info(
        "Ingesta completada el %s | fuente: UCI ML Repository (doi=%s)",
        datetime.now().isoformat(timespec="seconds"),
        metadata.get("doi"),
    )
    return output_path


def main() -> None:
    logger.info("=== Iniciando ingesta del dataset Adult / Census Income ===")
    features, targets, metadata = download_adult_dataset()
    df = build_raw_dataframe(features, targets)
    validate_minimum_shape(df)
    save_raw(df, metadata)
    logger.info("=== Ingesta finalizada con éxito ===")


if __name__ == "__main__":
    main()
