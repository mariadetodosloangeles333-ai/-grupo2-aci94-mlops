"""
build_features.py

Pipeline reutilizable de Feature Engineering para el proyecto ACI94
(Adult Income Dataset).

El mismo pipeline debe utilizarse durante el entrenamiento y durante
la inferencia en producción para evitar diferencias entre el procesamiento
del notebook y el procesamiento de la API.

Uso durante entrenamiento:

    from src.features.build_features import construir_pipeline_features

    pipeline_features = construir_pipeline_features()

    X_train_transformado = pipeline_features.fit_transform(X_train)
    X_test_transformado = pipeline_features.transform(X_test)

IMPORTANTE:
El pipeline debe ajustarse (fit) únicamente utilizando los datos de
entrenamiento. Nunca debe hacerse fit_transform() sobre test.
"""


from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer


# =============================================================================
# VERSIÓN DEL CONJUNTO DE FEATURES
# =============================================================================

FEATURE_SET_VERSION = "v1_baseline"


# =============================================================================
# CONFIGURACIÓN DEL DATASET
# =============================================================================

TARGET_COLUMN = "income"


# =============================================================================
# VARIABLES NUMÉRICAS
# =============================================================================
#
# education-num se conserva en lugar de education porque ambas variables
# representan información relacionada con el nivel educativo.
#
# Las variables numéricas serán:
#   1. Imputadas utilizando la mediana.
#   2. Estandarizadas utilizando StandardScaler.
# =============================================================================

NUMERIC_FEATURES = [
    "age",
    "education-num",
    "hours-per-week",
]


# =============================================================================
# VARIABLES NUMÉRICAS SESGADAS
# =============================================================================
#
# capital-gain y capital-loss presentan muchos valores iguales a cero y
# algunos valores extremadamente altos.
#
# Para cada variable se crean:
#
#   <variable>_flag
#       Indica si existe un valor positivo.
#
#   <variable>_log
#       Transformación logarítmica mediante log1p.
#
# Las variables originales son eliminadas después de la transformación.
# =============================================================================

SKEWED_NUMERIC_FEATURES = [
    "capital-gain",
    "capital-loss",
]


# Lista de variables generadas a partir de las variables sesgadas.

SKEWED_ENGINEERED_FEATURES = [
    feature
    for column in SKEWED_NUMERIC_FEATURES
    for feature in (
        f"{column}_flag",
        f"{column}_log",
    )
]


# =============================================================================
# VARIABLES CATEGÓRICAS
# =============================================================================

CATEGORICAL_FEATURES = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# =============================================================================
# PAÍSES FRECUENTES
# =============================================================================
#
# Esta lista es ESTÁTICA y debe estar respaldada por el análisis exploratorio
# realizado sobre el dataset.
#
# No se recalcula durante producción porque hacerlo podría provocar que
# diferentes batches generen estructuras de features diferentes.
# =============================================================================

FREQUENT_COUNTRIES = [
    "United-States",
    "Mexico",
    "Philippines",
    "Germany",
    "Canada",
    "Puerto-Rico",
    "El-Salvador",
    "India",
    "Cuba",
    "England",
    "Jamaica",
    "South",
]


# =============================================================================
# VARIABLES EXCLUIDAS
# =============================================================================
#
# education:
#     Se excluye porque education-num proporciona una representación
#     numérica del nivel educativo y evita mantener dos representaciones
#     de información muy similar.
#
# fnlwgt:
#     Representa un peso estadístico asociado al muestreo censal y no una
#     característica individual directa. Se excluye del modelo baseline
#     para mantener el conjunto de variables centrado en características
#     individuales interpretables.
# =============================================================================

EXCLUDED_FEATURES = [
    "education",
    "fnlwgt",
]


# =============================================================================
# VARIABLES OBLIGATORIAS
# =============================================================================

REQUIRED_FEATURES = (
    NUMERIC_FEATURES
    + SKEWED_NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# =============================================================================
# VALIDACIÓN DE COLUMNAS
# =============================================================================

def validar_columnas_entrada(
    df: pd.DataFrame,
) -> None:
    """
    Verifica que el dataframe contenga todas las variables requeridas
    por el pipeline.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe de entrada.

    Raises
    ------
    TypeError
        Si la entrada no es un DataFrame de pandas.

    ValueError
        Si faltan una o más variables obligatorias.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "La entrada debe ser un pandas DataFrame."
        )

    columnas_faltantes = [
        columna
        for columna in REQUIRED_FEATURES
        if columna not in df.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "El dataframe de entrada no contiene las variables "
            f"requeridas: {columnas_faltantes}"
        )


# =============================================================================
# LIMPIEZA DE VARIABLES CATEGÓRICAS
# =============================================================================

def limpiar_strings_categoricos(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Limpia las variables categóricas.

    Operaciones realizadas:
        - Elimina espacios al inicio y al final.
        - Convierte '?' en valores faltantes (NaN).
        - Conserva los valores NaN existentes.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe original.

    Returns
    -------
    pd.DataFrame
        Dataframe limpio.
    """

    df = df.copy()

    for columna in CATEGORICAL_FEATURES:

        if columna not in df.columns:
            continue

        # Se preservan los NaN reales.
        df[columna] = df[columna].apply(
            lambda valor: (
                np.nan
                if pd.isna(valor)
                else str(valor).strip()
            )
        )

        # El dataset utiliza '?' como representación de missing.
        df[columna] = df[columna].replace(
            "?",
            np.nan,
        )

    return df


# =============================================================================
# AGRUPACIÓN DE PAÍSES POCO FRECUENTES
# =============================================================================

def agrupar_paises_poco_frecuentes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrupa las categorías poco frecuentes de native-country en 'Other'.

    La lista de países frecuentes se encuentra definida de forma estática
    en FREQUENT_COUNTRIES.

    Los valores faltantes permanecen como NaN para que posteriormente sean
    tratados por el imputador de variables categóricas.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe previamente limpiado.

    Returns
    -------
    pd.DataFrame
        Dataframe con las categorías poco frecuentes agrupadas.
    """

    df = df.copy()

    columna = "native-country"

    if columna not in df.columns:
        return df

    valores_validos = df[columna].notna()

    df.loc[
        valores_validos
        & ~df.loc[
            valores_validos,
            columna,
        ].isin(FREQUENT_COUNTRIES),
        columna,
    ] = "Other"

    return df


# =============================================================================
# FEATURE ENGINEERING PARA VARIABLES SESGADAS
# =============================================================================

def transformar_variables_sesgadas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera nuevas variables para capital-gain y capital-loss.

    Para cada variable se generan:

        variable original
              |
              +----> indicador de presencia
              |
              +----> transformación log1p

    Las variables originales son eliminadas posteriormente.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe de entrada.

    Returns
    -------
    pd.DataFrame
        Dataframe con las nuevas variables.
    """

    df = df.copy()

    for columna in SKEWED_NUMERIC_FEATURES:

        if columna not in df.columns:
            continue

        # Convertir a numérico.
        #
        # Si aparece un valor no numérico, se convierte en NaN.
        valores_numericos = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

        # Indicador:
        # 1 = existe un valor positivo
        # 0 = no existe un valor positivo
        df[f"{columna}_flag"] = (
            valores_numericos.fillna(0) > 0
        ).astype(int)

        # Evitamos valores negativos antes de aplicar log1p.
        valores_seguras = valores_numericos.clip(
            lower=0
        )

        # Transformación logarítmica.
        df[f"{columna}_log"] = np.log1p(
            valores_seguras
        )

        # Eliminamos la variable original.
        df = df.drop(
            columns=[columna]
        )

    return df


# =============================================================================
# ELIMINACIÓN DE VARIABLES EXCLUIDAS
# =============================================================================

def eliminar_variables_excluidas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Elimina las variables que fueron excluidas del modelo baseline.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe de entrada.

    Returns
    -------
    pd.DataFrame
        Dataframe sin las variables excluidas.
    """

    df = df.copy()

    columnas_a_eliminar = [
        columna
        for columna in EXCLUDED_FEATURES
        if columna in df.columns
    ]

    return df.drop(
        columns=columnas_a_eliminar
    )


# =============================================================================
# PREPARACIÓN DEL DATAFRAME RAW
# =============================================================================

def preparar_dataframe_raw(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ejecuta las transformaciones determinísticas de Feature Engineering.

    Orden de procesamiento:

        1. Validación de columnas.
        2. Limpieza de variables categóricas.
        3. Agrupación de países poco frecuentes.
        4. Transformación de variables numéricas sesgadas.
        5. Eliminación de variables excluidas.

    Estas transformaciones no aprenden parámetros a partir de los datos.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe raw.

    Returns
    -------
    pd.DataFrame
        Dataframe preparado.
    """

    validar_columnas_entrada(df)

    df = limpiar_strings_categoricos(df)

    df = agrupar_paises_poco_frecuentes(df)

    df = transformar_variables_sesgadas(df)

    df = eliminar_variables_excluidas(df)

    return df


# =============================================================================
# CONSTRUCCIÓN DEL PIPELINE COMPLETO
# =============================================================================

def construir_pipeline_features() -> Pipeline:
    """
    Construye el pipeline completo y reutilizable de Feature Engineering.

    El pipeline contiene:

        Feature Engineering determinístico
                    +
        Imputación de valores faltantes
                    +
        Estandarización numérica
                    +
        One-Hot Encoding categórico

    El pipeline debe ajustarse (fit) únicamente utilizando el conjunto
    de entrenamiento.

    Ejemplo:

        pipeline_features = construir_pipeline_features()

        X_train_transformado = pipeline_features.fit_transform(X_train)

        X_test_transformado = pipeline_features.transform(X_test)

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline completo de Feature Engineering.
    """

    # -------------------------------------------------------------------------
    # TRANSFORMACIÓN DE VARIABLES NUMÉRICAS
    # -------------------------------------------------------------------------

    transformador_numerico = Pipeline(
        steps=[
            (
                "imputador",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "escalador",
                StandardScaler(),
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # TRANSFORMACIÓN DE VARIABLES CATEGÓRICAS
    # -------------------------------------------------------------------------

    transformador_categorico = Pipeline(
        steps=[
            (
                "imputador",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # COLUMN TRANSFORMER
    # -------------------------------------------------------------------------

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numerico",
                transformador_numerico,
                NUMERIC_FEATURES
                + SKEWED_ENGINEERED_FEATURES,
            ),
            (
                "categorico",
                transformador_categorico,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # -------------------------------------------------------------------------
    # PIPELINE COMPLETO
    # -------------------------------------------------------------------------
    #
    # FunctionTransformer permite incorporar nuestras transformaciones
    # determinísticas dentro del pipeline de scikit-learn.
    #
    # De esta forma, entrenamiento y producción utilizan exactamente
    # la misma secuencia de transformación.
    # -------------------------------------------------------------------------

    pipeline_features = Pipeline(
        steps=[
            (
                "preparar_raw",
                FunctionTransformer(
                    preparar_dataframe_raw,
                    validate=False,
                ),
            ),
            (
                "preprocesador",
                preprocesador,
            ),
        ]
    )

    return pipeline_features


# =============================================================================
# OBTENER NOMBRES DE FEATURES TRANSFORMADAS
# =============================================================================

def obtener_nombres_features(
    pipeline_ajustado: Pipeline,
) -> np.ndarray:
    """
    Obtiene los nombres de las variables después de todas las
    transformaciones.

    El pipeline debe haber sido previamente ajustado mediante fit().

    Parámetros
    ----------
    pipeline_ajustado : Pipeline
        Pipeline ya ajustado.

    Returns
    -------
    np.ndarray
        Nombres de las variables transformadas.
    """

    preprocesador = pipeline_ajustado.named_steps[
        "preprocesador"
    ]

    return preprocesador.get_feature_names_out()


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":

    RUTA_DATASET = "data/raw/adult.csv"

    print("=" * 70)
    print("ACI94 - Smoke Test de Feature Engineering")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Cargar dataset
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        RUTA_DATASET
    )

    print(
        f"Versión del Feature Set: {FEATURE_SET_VERSION}"
    )

    print(
        f"Dimensiones originales: {df.shape}"
    )

    # -------------------------------------------------------------------------
    # Verificar target
    # -------------------------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo '{TARGET_COLUMN}'."
        )

    # -------------------------------------------------------------------------
    # Separar variables predictoras y target
    # -------------------------------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    # -------------------------------------------------------------------------
    # Construir pipeline
    # -------------------------------------------------------------------------

    pipeline_features = construir_pipeline_features()

    # -------------------------------------------------------------------------
    # Smoke test
    #
    # IMPORTANTE:
    # Esto es solamente una prueba del pipeline.
    #
    # En el entrenamiento real, fit_transform() debe ejecutarse ÚNICAMENTE
    # sobre X_train después de separar train/test.
    # -------------------------------------------------------------------------

    X_transformado = pipeline_features.fit_transform(
        X
    )

    # -------------------------------------------------------------------------
    # Mostrar resultados
    # -------------------------------------------------------------------------

    print(
        f"Dimensiones del target: {y.shape}"
    )

    print(
        f"Dimensiones transformadas: {X_transformado.shape}"
    )

    print(
        f"Número de features generadas: "
        f"{X_transformado.shape[1]}"
    )

    print("=" * 70)
    print("Smoke test completado correctamente.")
    print("=" * 70)
