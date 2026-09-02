"""
Entrenamiento, evaluación y seguimiento reproducible de los modelos.

Se carga el dataset limpio, se reutiliza el pipeline de Feature
Engineering, se comparan distintos clasificadores mediante validación cruzada
estratificada y se analiza su comportamiento general y por subgrupos.
Los experimentos y sus resultados se registran en MLflow.
"""


# Parte 1: Importaciones y configuración general

# Importar librerías
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Ubicar la carpeta principal del repositorio.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Permitir importar los módulos del proyecto al ejecutar train.py directamente.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar los 2 feature set creados en feature engineering
from src.features.build_features import (  # noqa: E402
    FEATURE_SET_VERSION,
    FEATURE_SET_WITHOUT_SENSITIVE_VERSION,
    construir_pipeline_features,
)

# Ruta de los datos limpios
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "adult_clean.csv"

# Archivos principales que definen el entrenamiento
TRAINING_CODE_PATH = Path(__file__).resolve()
FEATURE_CODE_PATH = (
    PROJECT_ROOT / "src" / "features" / "build_features.py"
)

# Variable objetivo a predecir
TARGET_COLUMN = "income"

# Clases aprobadas para la variable objetivo
NEGATIVE_CLASS = "<=50K"
POSITIVE_CLASS = ">50K"
EXPECTED_CLASSES = {
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
}

# Variable utilizada para evaluar diferencias de comportamiento
SUBGROUP_COLUMN = "sex"
SUBGROUP_VALUES = ("Female", "Male")

# Variables conservadas para auditoría, pero excluidas de v2
SENSITIVE_COLUMNS = (
    "race",
    "sex",
    "native-country",
)

# Configuración reproducible de entrenamiento y evaluación
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Configuración del seguimiento de experimentos
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"
MLFLOW_EXPERIMENT_NAME = "adult-income-classification"

# Identificación inicial del dataset procesado
DATA_VERSION = "adult_clean_v1"

# Configura el almacenamiento local de experimentos y artefactos en mlflow
def configure_mlflow() -> None:

    # Crear la carpeta donde se guardarán modelos, gráficos y configuraciones
    MLFLOW_ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Utilizar SQLite para guardar los metadatos de MLflow
    tracking_uri = (
        f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)

    # Crear el experimento solamente si todavía no existe
    experiment = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    )

    if experiment is None:
        mlflow.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            artifact_location=(
                MLFLOW_ARTIFACTS_DIR.resolve().as_uri()
            ),
            tags={
                "problem_type": "binary_classification",
                "dataset": "Adult Census Income",
                "project": "ACI94",
            },
        )

    # Establecer el experimento activo
    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    print(
        f"MLflow configurado. Experimento: "
        f"{MLFLOW_EXPERIMENT_NAME}"
    )


# Calcula una huella única para identificar la versión exacta de los datos
def calculate_file_hash(file_path: Path) -> str:

    sha256 = hashlib.sha256()

    # Leer por bloques para no cargar todo el archivo en memoria
    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


# Obtiene el commit y verifica si existen cambios locales sin versionar
def get_git_information() -> dict[str, str]:

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        working_tree_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        return {
            "git_commit": commit,
            "git_dirty": str(bool(working_tree_status)),
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_commit": "not_available",
            "git_dirty": "not_available",
        }


# Organiza los parámetros generales y los hiperparámetros del modelo
def build_mlflow_parameters(
    experiment: dict,
) -> dict[str, object]:

    classifier = experiment["model"]

    # Parametros generales
    parameters = {
        "algorithm": experiment["algorithm"],
        "model_role": experiment["model_role"],
        "feature_set": experiment["feature_set"],
        "random_seed": RANDOM_SEED,
        "data_version": DATA_VERSION,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
    }

    # Registrar los hiperparámetros propios de cada algoritmo
    for parameter_name, parameter_value in (
        classifier.get_params(deep=False).items()
    ):
        parameters[
            f"classifier__{parameter_name}"
        ] = parameter_value

    return parameters


# Reúne la información que identifica los datos y el código utilizados
def build_mlflow_tags() -> dict[str, str]:

    git_information = get_git_information()

    return {
        "data_sha256": calculate_file_hash(DATA_PATH),
        "training_code_sha256": calculate_file_hash(
            TRAINING_CODE_PATH
        ),
        "feature_code_sha256": calculate_file_hash(
            FEATURE_CODE_PATH
        ),
        "git_commit": git_information["git_commit"],
        "git_dirty": git_information["git_dirty"],
        "evaluation_stage": "initial_comparison",
    }


# Crea la matriz de confusión para la etapa de evaluación indicada
def create_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: np.ndarray,
    run_name: str,
    evaluation_label: str = "OOF",
):

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
        display_labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )

    axis.set_title(
        f"Matriz de confusión {evaluation_label}\n"
        f"{run_name}"
    )
    figure.tight_layout()

    return figure


# Crea la curva ROC para la etapa de evaluación indicada
def create_roc_curve_figure(
    y_true: pd.Series,
    positive_probabilities: np.ndarray,
    run_name: str,
    evaluation_label: str = "OOF",
):

    y_true_binary = (
        y_true == POSITIVE_CLASS
    ).astype(int)

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    RocCurveDisplay.from_predictions(
        y_true_binary,
        positive_probabilities,
        name=run_name,
        ax=axis,
    )

    # Referencia correspondiente a una clasificación aleatoria
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Clasificador aleatorio",
    )

    axis.set_title(
        f"Curva ROC {evaluation_label}\n"
        f"{run_name}"
    )
    axis.legend(loc="lower right")
    figure.tight_layout()

    return figure


# Selecciona únicamente resultados numéricos válidos para MLflow
def build_mlflow_metrics(
    result: dict,
) -> dict[str, float]:

    metrics = {}

    for metric_name, metric_value in result.items():

        # Excluir nombres, versiones y demás valores de texto
        if not isinstance(
            metric_value,
            (int, float, np.integer, np.floating),
        ):
            continue

        numeric_value = float(metric_value)

        # MLflow debe recibir métricas numéricas finitas
        if np.isfinite(numeric_value):
            metrics[metric_name] = numeric_value

    return metrics


# Registra un experimento completo en MLflow
def log_experiment_to_mlflow(
    experiment: dict,
    pipeline: Pipeline,
    result: dict,
    X_model_train: pd.DataFrame,
    y_train: pd.Series,
    predicted_classes: np.ndarray,
    positive_probabilities: np.ndarray,
) -> None:

    parameters = build_mlflow_parameters(
        experiment
    )
    metrics = build_mlflow_metrics(
        result
    )
    tags = build_mlflow_tags()

    # Cada configuración se registra como un run independiente
    with mlflow.start_run(
        run_name=experiment["run_name"],
        tags=tags,
    ) as active_run:

        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        # Guardar un resumen auditable de la configuración y resultados
        mlflow.log_dict(
            {
                "parameters": parameters,
                "metrics": metrics,
                "tags": tags,
            },
            "configuration/run_summary.json",
        )

        # Registrar la matriz de confusión out-of-fold
        confusion_figure = create_confusion_matrix_figure(
            y_train,
            predicted_classes,
            experiment["run_name"],
        )

        try:
            mlflow.log_figure(
                confusion_figure,
                "plots/confusion_matrix_oof.png",
            )
        finally:
            plt.close(confusion_figure)

        # Registrar la curva ROC out-of-fold
        roc_figure = create_roc_curve_figure(
            y_train,
            positive_probabilities,
            experiment["run_name"],
        )

        try:
            mlflow.log_figure(
                roc_figure,
                "plots/roc_curve_oof.png",
            )
        finally:
            plt.close(roc_figure)

        # Ajustar el pipeline completo con todo el conjunto de entrenamiento
        fitted_pipeline = pipeline.fit(
            X_model_train,
            y_train,
        )

        input_example = (
            X_model_train
            .head(5)
            .copy()
        )

        # Representar las entradas numéricas como decimales en la firma
        # para permitir valores faltantes durante la inferencia
        integer_columns = input_example.select_dtypes(
            include=["integer"]
        ).columns

        input_example[integer_columns] = (
            input_example[integer_columns]
            .astype("float64")
        )

        example_predictions = fitted_pipeline.predict(
            input_example
        )

        model_signature = infer_signature(
            input_example,
            example_predictions,
        )

        # Guardar Feature Engineering y clasificador como un solo modelo
        mlflow.sklearn.log_model(
            sk_model=fitted_pipeline,
            name="model",
            signature=model_signature,
            input_example=input_example,
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        print(
            f"Run registrado en MLflow: "
            f"{active_run.info.run_id}"
        )


# Parte 2: Carga y validación del dataset limpio

def load_processed_data() -> pd.DataFrame:

    # El entrenamiento depende del resultado del pipeline de limpieza
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset limpio en: {DATA_PATH}. "
            "Ejecute primero: python src/cleaning/clean.py"
        )

    df = pd.read_csv(DATA_PATH)

    # Evitar iniciar el entrenamiento con un archivo vacío
    if df.empty:
        raise ValueError(
            "El dataset procesado no contiene observaciones."
        )

    # La variable objetivo debe existir
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo '{TARGET_COLUMN}'."
        )

    # La variable objetivo no debe tener faltantes
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"La variable objetivo '{TARGET_COLUMN}' contiene valores faltantes."
        )

    # Confirmar las clases de la variable objetivo
    detected_classes = set(
        df[TARGET_COLUMN].unique()
    )

    if detected_classes != EXPECTED_CLASSES:
        raise ValueError(
            "La variable objetivo no contiene exactamente las clases "
            f"esperadas. Clases detectadas: {sorted(detected_classes)}"
        )

    # Verificar que el análisis obligatorio por subgrupos sea posible
    if SUBGROUP_COLUMN not in df.columns:
        raise ValueError(
            f"No se encontró la variable de subgrupo "
            f"'{SUBGROUP_COLUMN}'."
        )
    
    # Los subgrupos no pueden tener faltantes
    if df[SUBGROUP_COLUMN].isna().any():
        raise ValueError(
            f"La variable de subgrupo "
            f"'{SUBGROUP_COLUMN}' contiene valores faltantes."
        )

    detected_subgroups = set(
        df[SUBGROUP_COLUMN].unique()
    )

    # La variable de subgrupos debe tener solo los valores esperados
    if detected_subgroups != set(SUBGROUP_VALUES):
        raise ValueError(
            "La variable de subgrupo no contiene exactamente "
            f"los valores esperados. Detectados: "
            f"{sorted(detected_subgroups)}"
        )

    print(
        f"Dataset procesado cargado: "
        f"{df.shape[0]} filas, {df.shape[1]} columnas."
    )

    return df


# Parte 3: División del dataset entre conjunto de entrenamiento y prueba

# Separar predictores y target con un test estratificado
def split_data(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    # Variables predictoras y variable objetivo
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # División estratificada para conservar aproximadamente las clases
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(
        f"Entrenamiento: {X_train.shape[0]} filas. "
        f"Prueba reservada: {X_test.shape[0]} filas."
    )

    return X_train, X_test, y_train, y_test


# Define las columnas que recibe cada versión del feature set
def select_model_input(
    X: pd.DataFrame,
    feature_set_version: str,
) -> pd.DataFrame:

    # Feature set sin variables sensibles
    if (
        feature_set_version
        == FEATURE_SET_WITHOUT_SENSITIVE_VERSION
    ):
        return X.drop(
            columns=list(SENSITIVE_COLUMNS)
        ).copy()

    return X.copy()


# Parte 4: Métricas de desempeño

# Función específica para G-Mean - métrica principal
def calculate_g_mean(
    y_true,
    y_pred,
) -> float:

    # Recall
    recall = recall_score(
        y_true,
        y_pred,
        pos_label=POSITIVE_CLASS,
        zero_division=0,
    )

    # Especificidad
    specificity = recall_score(
        y_true,
        y_pred,
        pos_label=NEGATIVE_CLASS,
        zero_division=0,
    )

    # G-Mean
    return float(
        np.sqrt(recall * specificity)
    )

# Otras métricas de desempeño
def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred,
    y_probability,
) -> dict[str, float]:

    # Fijar el orden para interpretar la matriz de confusión
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    )

    true_negatives, false_positives, false_negatives, true_positives = (
        matrix.ravel()
    )

    # Recall
    recall = recall_score(
        y_true,
        y_pred,
        pos_label=POSITIVE_CLASS,
        zero_division=0,
    )

    # Especificidad
    specificity_denominator = true_negatives + false_positives
    specificity = (
        true_negatives / specificity_denominator
        if specificity_denominator > 0
        else 0.0
    )

    # G-Mean
    g_mean = float(
        np.sqrt(recall * specificity)
    )

    # ROC-AUC
    y_true_binary = (
        y_true == POSITIVE_CLASS
    ).astype(int)

    # Resultados
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "g_mean": g_mean,
        "roc_auc": roc_auc_score(
            y_true_binary,
            y_probability,
        ),
        "true_negatives": float(true_negatives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "true_positives": float(true_positives),
    }

# Métricas utilizadas
# accuracy: es una referencia general pero no es confiable por el desbalance de clases
# balanced_accuracy: promedio del desempeño de ambas clases
# precision: confiabilidad de las predicciones positivas (>50K)
# recall: detección de los casos reales positivos (>50K)
# specificity: detección de los casos reales negativos (<=50K)
# F1: equilibrio entre precision y recall
# G-mean: equilibrio entre recall y especificidad
# ROC AUC: capacidad de separar y ordenar ambas clases
# TN, FP, FN, TP: base de la matriz de confusión


# Parte 5: Experimentos iniciales

# Modelos y configuraciones iniciales
def get_initial_experiments() -> list[dict]:

    return [
        # Modelo Dummy - siempre predice la clase mayoritaria
        # Se utiliza como referencia mínima
        {
            "run_name": "dummy-most-frequent", # nombre en mlflow
            "algorithm": "DummyClassifier", # familia del algoritmo
            "model_role": "minimum_baseline", # proposito del modelo en la comparación
            "model": DummyClassifier(
                strategy="most_frequent",
            ), # objeto con sus hiperparámetros
            "feature_set": FEATURE_SET_VERSION, # v1_baseline - variables sensibles
        },

        # Regresión sin balanceo de clases - para comparar el efecto de balancear las clases
        {
            "run_name": "logistic-regression-unbalanced",
            "algorithm": "LogisticRegression",
            "model_role": "linear_baseline",
            "model": LogisticRegression(
                C=1.0,
                class_weight=None,
                max_iter=1_000,
                random_state=RANDOM_SEED,
                solver="liblinear",
            ),
            "feature_set": FEATURE_SET_VERSION,
        },

        # Regresión con balanceo de clases - se busca observar el efecto de balancear las clases
        {
            "run_name": "logistic-regression-balanced",
            "algorithm": "LogisticRegression",
            "model_role": "imbalance_treatment",
            "model": LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1_000,
                random_state=RANDOM_SEED,
                solver="liblinear",
            ),
            "feature_set": FEATURE_SET_VERSION,
        },

        # Árbol de decisión - Familia no lineal
        {
            "run_name": "decision-tree-balanced",
            "algorithm": "DecisionTreeClassifier",
            "model_role": "nonlinear_baseline",
            "model": DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            "feature_set": FEATURE_SET_VERSION,
        },

        # Random forest - Ensamble de árboles
        {
            "run_name": "random-forest-balanced",
            "algorithm": "RandomForestClassifier",
            "model_role": "ensemble_candidate",
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            "feature_set": FEATURE_SET_VERSION,
        },

        # Regresión balanceada sin variables sensibles
        {
            "run_name": "logistic-regression-balanced-no-sensitive",
            "algorithm": "LogisticRegression",
            "model_role": "sensitive_feature_ablation",
            "feature_set": FEATURE_SET_WITHOUT_SENSITIVE_VERSION,
            "model": LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1_000,
                random_state=RANDOM_SEED,
                solver="liblinear",
            ),
        },

        # Árbol balanceado sin variables sensibles
        {
            "run_name": "decision-tree-balanced-no-sensitive",
            "algorithm": "DecisionTreeClassifier",
            "model_role": "sensitive_feature_ablation",
            "feature_set": FEATURE_SET_WITHOUT_SENSITIVE_VERSION,
            "model": DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        },

        # Random Forest balanceado sin variables sensibles
        {
            "run_name": "random-forest-balanced-no-sensitive",
            "algorithm": "RandomForestClassifier",
            "model_role": "sensitive_feature_ablation",
            "feature_set": FEATURE_SET_WITHOUT_SENSITIVE_VERSION,
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
        },
    ]


# Parte 6: Construcción del pipeline de modelado

# Conecta feature engineering y clasificador en un pipeline
def build_model_pipeline(
    classifier,
    feature_set_version: str = FEATURE_SET_VERSION,
) -> Pipeline:

    return Pipeline(
        steps=[
            # Se aplica el feature engineering
            (
                "features",
                construir_pipeline_features(
                    feature_set_version=feature_set_version,
                ),
            ),
            # Se incorpora el modelo
            (
                "classifier",
                classifier,
            ),
        ]
    )

# Integrar Feature Engineering y clasificador evita ajustar el
# preprocesamiento con información de las particiones de validación


# Parte 7: Validación cruzada y generalización

# Crea particiones reproducibles con la proporción de las clases
def create_stratified_cross_validation() -> StratifiedKFold:

    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


# Métricas calculadas en cada partición
def get_cross_validation_scoring() -> dict:

    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(
            precision_score,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "specificity": make_scorer(
            recall_score,
            pos_label=NEGATIVE_CLASS,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "g_mean": make_scorer(
            calculate_g_mean,
        ),
        "roc_auc": "roc_auc",
    }


# Evaluación de métricas, validación, estabilidad y tiempo
def evaluate_with_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:

    cross_validation = (
        create_stratified_cross_validation()
    )

    scoring = get_cross_validation_scoring()

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        scoring=scoring,
        cv=cross_validation,
        n_jobs=-1,
        return_train_score=True,
    )

    summary: dict[str, float] = {}

    for metric_name in scoring:
        train_values = scores[f"train_{metric_name}"]
        validation_values = scores[f"test_{metric_name}"]

        train_mean = float(
            np.mean(train_values)
        )

        validation_mean = float(
            np.mean(validation_values)
        )

        summary[f"train_{metric_name}_mean"] = train_mean
        summary[f"train_{metric_name}_std"] = float(
            np.std(train_values)
        )

        summary[f"cv_{metric_name}_mean"] = validation_mean
        summary[f"cv_{metric_name}_std"] = float(
            np.std(validation_values)
        )

        summary[f"gap_{metric_name}"] = (
            train_mean - validation_mean
        )

    summary["fit_time_mean"] = float(
        np.mean(scores["fit_time"])
    )

    summary["fit_time_std"] = float(
        np.std(scores["fit_time"])
    )

    summary["score_time_mean"] = float(
        np.mean(scores["score_time"])
    )

    return summary


# Parte 8: Predicciones out of fold

# Genera probabilidades y clases out of fold sobre el entrenamiento
def generate_out_of_fold_predictions(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:

    cross_validation = (
        create_stratified_cross_validation()
    )

    # Cada fila es predicha por un modelo que no fue ajustado con esa fila
    probability_matrix = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cross_validation,
        method="predict_proba",
        n_jobs=-1,
    )

    class_labels = sorted(
        y_train.unique()
    )

    positive_class_index = class_labels.index(
        POSITIVE_CLASS
    )

    positive_probabilities = probability_matrix[
        :,
        positive_class_index,
    ]

    # Aplicar el umbral estándar de clasificación binaria
    predicted_classes = np.where(
        positive_probabilities >= 0.5,
        POSITIVE_CLASS,
        NEGATIVE_CLASS,
    )

    return predicted_classes, positive_probabilities


# Parte 9: Evaluación por subgrupos

# Calcula métricas out-of-fold para los subgrupos Female y Male
def calculate_subgroup_metrics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    predicted_classes: np.ndarray,
    positive_probabilities: np.ndarray,
) -> dict[str, dict[str, float]]:

    # Validación de la variable
    if SUBGROUP_COLUMN not in X_train.columns:
        raise ValueError(
            f"No se encontró la variable de subgrupo "
            f"'{SUBGROUP_COLUMN}'."
        )

    subgroup_results = {}

    for subgroup in SUBGROUP_VALUES:
        subgroup_mask = (
            X_train[SUBGROUP_COLUMN]
            .eq(subgroup)
            .to_numpy()
        )

        # Validación de los valores
        if not subgroup_mask.any():
            raise ValueError(
                f"El subgrupo '{subgroup}' no contiene observaciones."
            )

        y_subgroup = y_train.iloc[
            np.flatnonzero(subgroup_mask)
        ]

        # Validación de las clases
        if y_subgroup.nunique() < 2:
            raise ValueError(
                f"El subgrupo '{subgroup}' no contiene ambas clases "
                "y no permite calcular ROC AUC."
            )

        predicted_subgroup = predicted_classes[
            subgroup_mask
        ]

        probabilities_subgroup = positive_probabilities[
            subgroup_mask
        ]

        metrics = calculate_classification_metrics(
            y_subgroup,
            predicted_subgroup,
            probabilities_subgroup,
        )

        # Representación del grupo
        metrics["sample_count"] = float(
            subgroup_mask.sum()
        )

        # Proporción real de positivos
        metrics["actual_positive_rate"] = float(
            (y_subgroup == POSITIVE_CLASS).mean()
        )

        # Predicción de positivos
        metrics["predicted_positive_rate"] = float(
            (
                predicted_subgroup == POSITIVE_CLASS
            ).mean()
        )

        subgroup_results[subgroup] = metrics

    return subgroup_results


# Organiza las métricas por subgrupo y calcula sus diferencias
def summarize_subgroup_metrics(
    subgroup_results: dict[str, dict[str, float]],
) -> dict[str, float]:

    summary = {}

    for subgroup, metrics in subgroup_results.items():
        subgroup_name = subgroup.lower()

        for metric_name, metric_value in metrics.items():
            summary[
                f"subgroup_{subgroup_name}_{metric_name}"
            ] = float(metric_value)

    # Medir diferencias de desempeño sin asignarles todavía una causa
    metrics_to_compare = [
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "g_mean",
        "roc_auc",
        "predicted_positive_rate",
    ]

    first_subgroup, second_subgroup = SUBGROUP_VALUES

    for metric_name in metrics_to_compare:
        first_value = subgroup_results[
            first_subgroup
        ][metric_name]

        second_value = subgroup_results[
            second_subgroup
        ][metric_name]

        summary[f"subgroup_gap_{metric_name}"] = float(
            abs(first_value - second_value)
        )

    return summary


# Parte 10: Ejecución de los primeros experimentos

# Ejecuta y evalúa los modelos
def run_initial_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[dict]:

    comparison_results = []

    # Evaluar cada configuración bajo las mismas particiones y métricas
    for experiment in get_initial_experiments():
        run_name = experiment["run_name"]
        classifier = experiment["model"]
        feature_set_version = experiment["feature_set"]
        X_model_train = select_model_input(
            X_train,
            feature_set_version,
        )

        print(
            f"\nEvaluando: {run_name} "
            f"| Feature set: {feature_set_version}"
        )

        pipeline = build_model_pipeline(
            classifier,
            feature_set_version=feature_set_version,
        )

        cv_metrics = evaluate_with_cross_validation(
            pipeline,
            X_model_train,
            y_train,
        )

        predicted_classes, positive_probabilities = (
            generate_out_of_fold_predictions(
                pipeline,
                X_model_train,
                y_train,
            )
        )

        oof_metrics = calculate_classification_metrics(
            y_train,
            predicted_classes,
            positive_probabilities,
        )

        oof_summary = {
            f"oof_{metric_name}": float(metric_value)
            for metric_name, metric_value in oof_metrics.items()
        }

        subgroup_results = calculate_subgroup_metrics(
            X_train,
            y_train,
            predicted_classes,
            positive_probabilities,
        )

        subgroup_summary = summarize_subgroup_metrics(
            subgroup_results
        )

        result = {
            "run_name": run_name,
            "algorithm": experiment["algorithm"],
            "model_role": experiment["model_role"],
            "feature_set": feature_set_version,
            **cv_metrics,
            **oof_summary,
            **subgroup_summary,
        }

        log_experiment_to_mlflow(
            experiment=experiment,
            pipeline=pipeline,
            result=result,
            X_model_train=X_model_train,
            y_train=y_train,
            predicted_classes=predicted_classes,
            positive_probabilities=positive_probabilities,
        )

        comparison_results.append(result)

    return comparison_results


# Parte 11: Organización de resultados

# Convierte los resultados en una tabla ordenada por G-Mean
def create_comparison_table(
    comparison_results: list[dict],
) -> pd.DataFrame:

    # Validación de resultados
    if not comparison_results:
        raise ValueError(
            "No existen resultados para construir la comparación."
        )

    results_df = pd.DataFrame(comparison_results)

    results_df = (
        results_df
        .sort_values(
            by="cv_g_mean_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return results_df


# Parte 12: Presentación de resultados

# Desempeño general, subgrupos y tasas de predicción
def print_comparison_report(
    comparison_table: pd.DataFrame,
) -> None:
    general_columns = [
        "run_name",
        "feature_set",
        "train_g_mean_mean",
        "cv_g_mean_mean",
        "gap_g_mean",
        "cv_g_mean_std",
        "cv_f1_mean",
        "cv_recall_mean",
        "cv_specificity_mean",
        "cv_roc_auc_mean",
        "fit_time_mean",
    ]

    print("\nResultados generales de validación cruzada:")
    print(
        comparison_table[general_columns]
        .round(4)
        .to_string(index=False)
    )

    # La composición real es igual para todos los modelos
    reference_result = comparison_table.iloc[0]

    print("\nComposición real de los subgrupos:")
    print(
        "Female: "
        f"{int(reference_result['subgroup_female_sample_count'])} casos, "
        "tasa real >50K = "
        f"{reference_result['subgroup_female_actual_positive_rate']:.4f}"
    )

    print(
        "Male: "
        f"{int(reference_result['subgroup_male_sample_count'])} casos, "
        "tasa real >50K = "
        f"{reference_result['subgroup_male_actual_positive_rate']:.4f}"
    )

    subgroup_columns = [
        "run_name",
        "feature_set",
        "subgroup_female_g_mean",
        "subgroup_male_g_mean",
        "subgroup_gap_g_mean",
        "subgroup_female_recall",
        "subgroup_male_recall",
        "subgroup_gap_recall",
        "subgroup_female_specificity",
        "subgroup_male_specificity",
        "subgroup_gap_specificity",
    ]

    print("\nResultados por subgrupo:")
    print(
        comparison_table[subgroup_columns]
        .round(4)
        .to_string(index=False)
    )

    rate_columns = [
        "run_name",
        "feature_set",
        "subgroup_female_predicted_positive_rate",
        "subgroup_male_predicted_positive_rate",
    ]

    print("\nTasas de predicción positiva por subgrupo:")
    print(
        comparison_table[rate_columns]
        .round(4)
        .to_string(index=False)
    )


# Parte 13: Ejecución principal

def main() -> None:

    print("=" * 70)
    print("ACI94 - Comparación inicial de modelos")
    print("=" * 70)

    configure_mlflow()

    df = load_processed_data()

    X_train, X_test, y_train, y_test = split_data(df)

    comparison_results = run_initial_comparison(
        X_train,
        y_train,
    )

    comparison_table = create_comparison_table(
        comparison_results,
    )

    print_comparison_report(
        comparison_table
    )

    print("\nConjunto de prueba reservado:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")


if __name__ == "__main__":
    main()

"""
Conclusión de esta etapa:

Se debe seleccionar v2_without_sensitive (la versión sin las variables sensibles) para las
siguientes etapas ya que excluir las variables de sex, race y native country mantiene
practicamente el mismo desempeño predictivo general, pero además reduce varias diferencias
de comportamiento al realizar el análisis de los subgrupos (Female y Male) en las métricas
de recall, especificidad y tasa de predicción positiva, se mejora el G-Mean del subgrupo 
de mujeres y el de subgrupo de hombres permance igual. Por lo tanto, solo se conservaran 
estas variables sensibles para auditoria y no se utilizaran como variables predictoras.

También es importante mencionar que se realizó un modelo Dummy para tomar como punto de 
referencia mínima y una regresión logística no balanceada para observar el efecto del 
tratamiento del desbalance, los modelos con class_weight="balanced" mejoraron bastante en 
comparación con estas referencias, por lo que se considero que el tratamiento del desbalance 
fue efectivo.

Finalmente, las brechas pequeñas entre entrenamiento y validación no muestran evidencia 
importante de sobreajuste en los modelos prometedores. La selección definitiva se realizará 
después de la búsqueda de hiperparámetros y la evaluación final con el conjunto de test,
los modelos prometedores son regresión logística, árbol de decisión y random forest (todos con
class_weight="balanced").
"""

