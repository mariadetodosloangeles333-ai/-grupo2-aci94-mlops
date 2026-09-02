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

from pathlib import Path

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

# Versión baseline que utiliza todas las variables aprobadas
FEATURE_SET_VERSION = "v1_baseline"

# Versión alternativa que excluye variables sensibles de la predicción (sex, race y native country)
FEATURE_SET_WITHOUT_SENSITIVE_VERSION = "v2_without_sensitive"

# Versiones de Feature Engineering admitidas por el pipeline
SUPPORTED_FEATURE_SET_VERSIONS = (
    FEATURE_SET_VERSION,
    FEATURE_SET_WITHOUT_SENSITIVE_VERSION,
)


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

# Variables disponibles para auditoría y evaluación de posibles sesgos
SENSITIVE_FEATURES = [
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
#     característica individual directa. Se excluye del modelo
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


def obtener_configuracion_features(
    feature_set_version: str,
) -> tuple[list[str], list[str]]:
    """
    Obtiene las variables categóricas y obligatorias de cada feature set.
    """
    if feature_set_version not in SUPPORTED_FEATURE_SET_VERSIONS:
        raise ValueError(
            "Versión de features no soportada: "
            f"'{feature_set_version}'. Versiones permitidas: "
            f"{SUPPORTED_FEATURE_SET_VERSIONS}"
        )

    if feature_set_version == FEATURE_SET_VERSION:
        categorical_features = CATEGORICAL_FEATURES.copy()

    else:
        categorical_features = [
            feature
            for feature in CATEGORICAL_FEATURES
            if feature not in SENSITIVE_FEATURES
        ]

    required_features = (
        NUMERIC_FEATURES
        + SKEWED_NUMERIC_FEATURES
        + categorical_features
    )

    return categorical_features, required_features

# =============================================================================
# VALIDACIÓN DE COLUMNAS
# =============================================================================

def validar_columnas_entrada(
    df: pd.DataFrame,
    required_features: list[str] | None = None,
) -> None:
    """
    Verifica que el dataframe contenga todas las variables requeridas
    por el pipeline.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe de entrada.

    required_features : list[str] | None
        Variables que la versión del feature set necesita como entrada.

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

    # Utilizar las variables del baseline cuando no se indique otra versión
    if required_features is None:
        required_features = REQUIRED_FEATURES

    columnas_faltantes = [
        columna
        for columna in required_features
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
    categorical_features: list[str] | None = None,
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

    categorical_features : list[str] | None
        Variables categóricas que deben limpiarse.

    Returns
    -------
    pd.DataFrame
        Dataframe limpio.
    """

    df = df.copy()

    # Utilizar las categóricas del baseline si no se indica otra versión
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    for columna in categorical_features:

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
    Elimina las variables que fueron excluidas de los feature sets.

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
# PREPARACIÓN DEL DATAFRAME
# =============================================================================

def preparar_dataframe(
    df: pd.DataFrame,
    feature_set_version: str = FEATURE_SET_VERSION,
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
        Dataframe.

    feature_set_version : str
        Versión de Feature Engineering que debe aplicarse.

    Returns
    -------
    pd.DataFrame
        Dataframe preparado.
    """

    categorical_features, required_features = (
        obtener_configuracion_features(
            feature_set_version
        )
    )

    validar_columnas_entrada(
        df,
        required_features=required_features,
    )

    df = limpiar_strings_categoricos(
        df,
        categorical_features=categorical_features,
    )

    if "native-country" in categorical_features:
        df = agrupar_paises_poco_frecuentes(df)

    df = transformar_variables_sesgadas(df)

    df = eliminar_variables_excluidas(df)

    return df


# =============================================================================
# CONSTRUCCIÓN DEL PIPELINE COMPLETO
# =============================================================================

def construir_pipeline_features(
    feature_set_version: str = FEATURE_SET_VERSION,
) -> Pipeline:
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

    categorical_features, _ = (
        obtener_configuracion_features(
            feature_set_version
        )
    )

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
                categorical_features,
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
                "preparar_dataframe",
                FunctionTransformer(
                    preparar_dataframe,
                    validate=False,
                    kw_args={
                        "feature_set_version": feature_set_version,
                    },
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

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    RUTA_DATASET = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "adult_clean.csv"
    )

    print("=" * 70)
    print("ACI94 - Smoke Test de Feature Engineering")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Cargar dataset
    # -------------------------------------------------------------------------

    if not RUTA_DATASET.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset procesado en: "
            f"{RUTA_DATASET}. Ejecute primero el pipeline de limpieza."
        )

    df = pd.read_csv(
        RUTA_DATASET
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

    # -------------------------------------------------------------------------
    # Smoke test
    #
    # IMPORTANTE:
    # Esto es solamente una prueba técnica del pipeline.
    #
    # En el entrenamiento real, fit_transform() debe ejecutarse únicamente
    # sobre X_train después de separar entrenamiento y prueba.
    # -------------------------------------------------------------------------

    # Probar todas las versiones admitidas del Feature Engineering
    for feature_set_version in SUPPORTED_FEATURE_SET_VERSIONS:

        print("-" * 70)
        print(
            f"Versión del Feature Set: "
            f"{feature_set_version}"
        )

        pipeline_features = construir_pipeline_features(
            feature_set_version=feature_set_version,
        )

        if (
            feature_set_version
            == FEATURE_SET_WITHOUT_SENSITIVE_VERSION
        ):
            X_version = X.drop(
                columns=SENSITIVE_FEATURES
            )

        else:
            X_version = X

        # Ajustar únicamente como prueba técnica del pipeline
        X_transformado = pipeline_features.fit_transform(
            X_version
        )

        feature_names = obtener_nombres_features(
            pipeline_features
        )

        sensitive_feature_names = [
            feature_name
            for feature_name in feature_names
            if any(
                f"__{sensitive_feature}_"
                in feature_name
                for sensitive_feature in SENSITIVE_FEATURES
            )
        ]

        print(
            f"Dimensiones originales: {X_version.shape}"
        )

        print(
            f"Dimensiones del target: {y.shape}"
        )

        print(
            f"Dimensiones transformadas: "
            f"{X_transformado.shape}"
        )

        print(
            f"Número de features generadas: "
            f"{X_transformado.shape[1]}"
        )

        print(
            f"Features sensibles generadas: "
            f"{len(sensitive_feature_names)}"
        )

        if (
            feature_set_version
            == FEATURE_SET_WITHOUT_SENSITIVE_VERSION
            and sensitive_feature_names
        ):
            raise ValueError(
                "La versión sin variables sensibles generó "
                f"features no permitidas: "
                f"{sensitive_feature_names}"
            )

    print("=" * 70)
    print("Smoke test completado correctamente.")
    print("=" * 70)
