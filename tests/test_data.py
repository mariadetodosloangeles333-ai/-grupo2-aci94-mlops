"""
test_data.py

Pruebas de calidad de datos para el proyecto ACI94 (Adult Income Dataset).

Ejecutar con:
    pytest tests/test_data.py -v
"""

import pandas as pd
import pytest
import os

# Ruta generada por src/ingestion/ingest.py (ver save_raw()).
DATA_PATH = "data/raw/adult_raw.csv"

# Columnas que el dataset SIEMPRE debe tener (esquema mínimo esperado)
EXPECTED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

# Columnas que NUNCA deberían tener nulos "reales" después de la ingesta
# (aunque en crudo puedan venir codificados como "?", eso se valida aparte)
REQUIRED_NON_NULL_COLUMNS = ["age", "income"]


@pytest.fixture(scope="module")
def raw_df():
    """Carga el dataset una sola vez para todas las pruebas de este archivo."""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"No se encontró el archivo de datos en {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------------
# ESQUEMA
# ---------------------------------------------------------------------------
def test_columns_exist(raw_df):
    """Todas las columnas esperadas deben estar presentes en el dataset."""
    missing_cols = set(EXPECTED_COLUMNS) - set(raw_df.columns)
    assert not missing_cols, f"Faltan columnas esperadas: {missing_cols}"


def test_no_unexpected_extra_columns(raw_df):
    """
    Alerta si aparecen columnas nuevas no contempladas (podría indicar un
    cambio de esquema en la fuente de datos, algo que el equipo debe revisar).
    """
    extra_cols = set(raw_df.columns) - set(EXPECTED_COLUMNS)
    assert not extra_cols, f"Aparecieron columnas no esperadas: {extra_cols}"


def test_minimum_row_count(raw_df):
    """El dataset debe tener un volumen mínimo razonable de registros."""
    assert raw_df.shape[0] > 1000, (
        f"El dataset tiene muy pocas filas ({raw_df.shape[0]}), "
        "podría estar incompleto o mal cargado."
    )


# ---------------------------------------------------------------------------
# TIPOS DE DATOS
# ---------------------------------------------------------------------------
def test_numeric_columns_are_numeric(raw_df):
    """age, education-num y hours-per-week deben ser numéricas."""
    numeric_cols = ["age", "education-num", "hours-per-week",
                     "capital-gain", "capital-loss"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(raw_df[col]), (
            f"La columna '{col}' debería ser numérica, "
            f"pero tiene tipo {raw_df[col].dtype}"
        )


# ---------------------------------------------------------------------------
# RANGOS VÁLIDOS
# ---------------------------------------------------------------------------
def test_age_within_valid_range(raw_df):
    """La edad debe estar en un rango humano razonable (16 a 100 años)."""
    assert raw_df["age"].between(16, 100).all(), (
        "Hay valores de 'age' fuera del rango esperado (16-100)."
    )


def test_hours_per_week_within_valid_range(raw_df):
    """Las horas trabajadas por semana no deberían superar un máximo lógico."""
    assert raw_df["hours-per-week"].between(1, 99).all(), (
        "Hay valores de 'hours-per-week' fuera del rango esperado (1-99)."
    )


def test_capital_gain_loss_non_negative(raw_df):
    """capital-gain y capital-loss no deberían tener valores negativos."""
    assert (raw_df["capital-gain"] >= 0).all(), "capital-gain tiene valores negativos"
    assert (raw_df["capital-loss"] >= 0).all(), "capital-loss tiene valores negativos"


def test_income_only_expected_categories(raw_df):
    """La variable objetivo income solo debe tener las 2 categorías esperadas."""
    # Se normalizan espacios porque el dataset original trae valores como ' <=50K'
    unique_values = set(raw_df["income"].astype(str).str.strip())
    expected_values = {"<=50K", ">50K", "<=50K.", ">50K."}  # variantes con punto en test set
    unexpected = unique_values - expected_values
    assert not unexpected, f"Valores inesperados en income: {unexpected}"


# ---------------------------------------------------------------------------
# MISSING VALUES / VARIABLES OBLIGATORIAS
# ---------------------------------------------------------------------------
def test_required_columns_have_no_nulls(raw_df):
    """Las columnas obligatorias no deben tener valores nulos."""
    for col in REQUIRED_NON_NULL_COLUMNS:
        null_count = raw_df[col].isna().sum()
        assert null_count == 0, (
            f"La columna obligatoria '{col}' tiene {null_count} valores nulos."
        )


def test_missing_values_ratio_below_threshold(raw_df):
    """
    Ninguna columna debería tener más de un 50% de valores faltantes
    (considerando también los '?' codificados como missing).
    """
    df_check = raw_df.replace("?", pd.NA)
    missing_ratio = df_check.isna().mean()
    columns_too_empty = missing_ratio[missing_ratio > 0.5]
    assert columns_too_empty.empty, (
        f"Columnas con más de 50% de valores faltantes: "
        f"{columns_too_empty.to_dict()}"
    )


def test_duplicated_rows_below_threshold(raw_df):
    """El porcentaje de filas duplicadas no debería ser excesivo."""
    dup_ratio = raw_df.duplicated().mean()
    assert dup_ratio < 0.05, (
        f"Hay un {dup_ratio:.2%} de filas duplicadas, por encima del umbral de 5%."
    )
