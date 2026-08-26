"""
build_features.py

Pipeline reutilizable de Feature Engineering para el proyecto ACI94 (Adult Income Dataset).

Este mismo pipeline debe usarse tanto en el notebook de exploración/entrenamiento
como en producción (API de inferencia), para evitar el problema de tener
"Notebook Feature Engineering" y "Production Feature Engineering" por separado.

Uso típico:
    from src.features.build_features import build_feature_pipeline

    pipeline = build_feature_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)   # NUNCA fit en test (evita leakage)
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Versión del set de features (loguear este valor en MLflow como parámetro
# "feature_set" para trazabilidad de cada run del experimento).
# ---------------------------------------------------------------------------
FEATURE_SET_VERSION = "v1_baseline"

# ---------------------------------------------------------------------------
# Definición de columnas según el diagnóstico hecho en el EDA
# ---------------------------------------------------------------------------

# Variables numéricas "normales" -> se escalan
NUMERIC_FEATURES = [
    "age",
    "education-num",   # se usa esta y se descarta 'education' (son redundantes)
    "hours-per-week",
]

# Variables numéricas con distribución muy sesgada (la mayoría en cero)
# -> se tratan aparte con un flag binario + log1p
SKEWED_NUMERIC_FEATURES = [
    "capital-gain",
    "capital-loss",
]

# Variables categóricas nominales -> One-Hot Encoding
CATEGORICAL_FEATURES = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Columnas que se EXCLUYEN explícitamente del modelo (con su justificación):
#   - "education"   : redundante con education-num (misma información, ya numérica)
#   - "fnlwgt"      : es un peso estadístico censal, no una característica de
#                     la persona; no aporta señal predictiva real y podría
#                     introducir ruido/leakage conceptual.
EXCLUDED_FEATURES = ["education", "fnlwgt"]

TARGET_COLUMN = "income"


# ---------------------------------------------------------------------------
# Función auxiliar: agrupa categorías poco frecuentes de native-country
# ---------------------------------------------------------------------------
def group_rare_countries(df: pd.DataFrame, column: str = "native-country",
                          min_freq: int = 100) -> pd.DataFrame:
    """
    Agrupa países con muy pocas observaciones en la categoría 'Other'
    para reducir la alta cardinalidad de native-country.
    """
    df = df.copy()
    counts = df[column].value_counts()
    rare_categories = counts[counts < min_freq].index
    df[column] = df[column].replace(rare_categories, "Other")
    return df


# ---------------------------------------------------------------------------
# Función auxiliar: limpia strings (quita espacios y estandariza "?")
# ---------------------------------------------------------------------------
def clean_categorical_strings(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    El dataset original trae valores con espacios (' Private') y los
    faltantes codificados como '?'. Esto los normaliza a NaN para que
    el imputer los trate correctamente.
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("?", np.nan)
    return df


# ---------------------------------------------------------------------------
# Transformador para las variables numéricas sesgadas (capital-gain/loss)
# ---------------------------------------------------------------------------
def transform_skewed_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Para cada columna sesgada crea:
      - un flag binario que indica si el valor es distinto de cero
      - una versión log1p del valor original (reduce el efecto de outliers)
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_flag"] = (df[col] != 0).astype(int)
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
    return df.drop(columns=columns)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def build_feature_pipeline() -> ColumnTransformer:
    """
    Construye el ColumnTransformer que se debe ajustar (fit) SOLO con el
    set de entrenamiento, y luego aplicar (transform) sobre validación/test
    y sobre cualquier input nuevo en producción (API).
    """

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Las columnas *_flag y *_log generadas por transform_skewed_numeric
    # se tratan como numéricas normales dentro de este ColumnTransformer.
    skewed_output_columns = []
    for col in SKEWED_NUMERIC_FEATURES:
        skewed_output_columns += [f"{col}_flag", f"{col}_log"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES + skewed_output_columns),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # cualquier columna no listada explícitamente se descarta
    )

    return preprocessor


def prepare_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las transformaciones "manuales" (no aprendidas de los datos)
    ANTES de pasar el dataframe al ColumnTransformer:
        1. Limpieza de strings y '?' -> NaN
        2. Agrupación de países poco frecuentes
        3. Generación de flags/log para variables sesgadas
        4. Eliminación de columnas excluidas (education, fnlwgt)

    Esta función debe llamarse igual en entrenamiento y en la API.
    """
    df = df.copy()
    df = clean_categorical_strings(df, CATEGORICAL_FEATURES)
    df = group_rare_countries(df, column="native-country", min_freq=100)
    df = transform_skewed_numeric(df, SKEWED_NUMERIC_FEATURES)
    df = df.drop(columns=[c for c in EXCLUDED_FEATURES if c in df.columns])
    return df


if __name__ == "__main__":
    # Prueba rápida / smoke test manual.
    # Ajusta la ruta según donde tengan el dataset crudo procesado por ingest.py
    raw_path = "data/raw/adult.csv"
    df = pd.read_csv(raw_path)

    df_prepared = prepare_raw_dataframe(df)
    X = df_prepared.drop(columns=[TARGET_COLUMN])
    y = df_prepared[TARGET_COLUMN]

    pipeline = build_feature_pipeline()
    X_transformed = pipeline.fit_transform(X)

    print(f"Feature set version: {FEATURE_SET_VERSION}")
    print(f"Shape original: {df.shape}")
    print(f"Shape transformado: {X_transformed.shape}")
