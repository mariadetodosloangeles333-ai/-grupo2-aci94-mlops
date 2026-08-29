"""
Evaluación final del modelo seleccionado y registro en MLflow.

El modelo se entrena con el conjunto de entrenamiento y se evalúa una única
vez con el conjunto de test reservado. También se analiza su comportamiento
por subgrupos y se generan los artefactos finales.
"""


# Parte 1: Importaciones y configuración general

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier

# Ubicar la carpeta principal del repositorio
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Permitir ejecutar evaluate.py directamente
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar feature set sin las variables sensibles
from src.features.build_features import (  # noqa: E402
    FEATURE_SET_WITHOUT_SENSITIVE_VERSION,
)

# Importar todo lo necesario de train.py
from src.training.train import (  # noqa: E402
    POSITIVE_CLASS,
    RANDOM_SEED,
    build_model_pipeline,
    calculate_classification_metrics,
    calculate_subgroup_metrics,
    summarize_subgroup_metrics,
    DATA_VERSION,
    TEST_SIZE,
    build_mlflow_metrics,
    build_mlflow_tags,
    calculate_file_hash,
    create_confusion_matrix_figure,
    create_roc_curve_figure,
    configure_mlflow,
    load_processed_data,
    select_model_input,
    split_data,
)

# Feature set seleccionado
SELECTED_FEATURE_SET = FEATURE_SET_WITHOUT_SENSITIVE_VERSION

# Identificación del modelo y del run que produjo la selección
SELECTED_ALGORITHM = "RandomForestClassifier"
SELECTED_TUNING_RUN_ID = (
    "ed15bec223ef45b7b4d45d66312374e8"
)

# Resultado de validación utilizado como referencia
VALIDATION_G_MEAN = 0.8338

# Criterios definidos antes de evaluar el conjunto de test
# Se definieron considerando los resultados de validación
MINIMUM_TEST_G_MEAN = 0.80
MAXIMUM_G_MEAN_DIFFERENCE = 0.03
MINIMUM_TEST_RECALL = 0.75
MINIMUM_TEST_SPECIFICITY = 0.75

# Nombre que se utilizará posteriormente en Model Registry
REGISTERED_MODEL_NAME = "adult-income-classifier"


# Parte 2: Construcción del modelo seleccionado

# Reconstruye el pipeline ganador con el feature set y los
# hiperparámetros seleccionados mediante validación cruzada
def build_selected_pipeline():

    # Hiperparámetros ganadores
    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    return build_model_pipeline(
        classifier=classifier,
        feature_set_version=SELECTED_FEATURE_SET,
    )


# Parte 3: Evaluación final sobre el conjunto de test

# Calcula las métricas generales y por subgrupo en el test reservado
def evaluate_final_pipeline(
    fitted_pipeline,
    X_model_test: pd.DataFrame,
    X_audit_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[
    dict[str, float],
    np.ndarray,
    np.ndarray,
]:

    predicted_classes = fitted_pipeline.predict(
        X_model_test
    )

    probability_matrix = fitted_pipeline.predict_proba(
        X_model_test
    )

    class_labels = list(
        fitted_pipeline
        .named_steps["classifier"]
        .classes_
    )

    positive_class_index = class_labels.index(
        POSITIVE_CLASS
    )

    positive_probabilities = probability_matrix[
        :,
        positive_class_index,
    ]

    general_metrics = calculate_classification_metrics(
        y_test,
        predicted_classes,
        positive_probabilities,
    )

    general_summary = {
        f"test_{metric_name}": float(metric_value)
        for metric_name, metric_value
        in general_metrics.items()
    }

    subgroup_results = calculate_subgroup_metrics(
        X_audit_test,
        y_test,
        predicted_classes,
        positive_probabilities,
    )

    subgroup_summary = {
        f"test_{metric_name}": float(metric_value)
        for metric_name, metric_value
        in summarize_subgroup_metrics(
            subgroup_results
        ).items()
    }

    # Comparar validación y test sin utilizar el test para reajustar
    generalization_summary = {
        "validation_g_mean": VALIDATION_G_MEAN,
        "validation_test_g_mean_difference": float(
            VALIDATION_G_MEAN
            - general_metrics["g_mean"]
        ),
        "validation_test_g_mean_absolute_difference": float(
            abs(
                VALIDATION_G_MEAN
                - general_metrics["g_mean"]
            )
        ),
    }

    evaluation_summary = {
        **general_summary,
        **subgroup_summary,
        **generalization_summary,
    }

    return (
        evaluation_summary,
        predicted_classes,
        positive_probabilities,
    )


# El modelo solo se considera candidato si supera todos
# los criterios de desempeño y generalización establecidos anteriormente
def validate_final_model(
    evaluation_summary: dict[str, float],
) -> tuple[bool, dict[str, bool]]:

    validation_checks = {
        "minimum_test_g_mean": (
            evaluation_summary["test_g_mean"]
            >= MINIMUM_TEST_G_MEAN
        ),
        "maximum_g_mean_difference": (
            evaluation_summary[
                "validation_test_g_mean_absolute_difference"
            ]
            <= MAXIMUM_G_MEAN_DIFFERENCE
        ),
        "minimum_test_recall": (
            evaluation_summary["test_recall"]
            >= MINIMUM_TEST_RECALL
        ),
        "minimum_test_specificity": (
            evaluation_summary["test_specificity"]
            >= MINIMUM_TEST_SPECIFICITY
        ),
    }

    validation_passed = all(
        validation_checks.values()
    )

    return validation_passed, validation_checks


# Parte 4: Importancia de features

# La importancia del Random Forest mide la contribución predictiva
# dentro del modelo pero no demuestra relaciones causales
def create_feature_importance_table(
    fitted_pipeline,
) -> pd.DataFrame:

    feature_pipeline = fitted_pipeline.named_steps[
        "features"
    ]

    preprocessor = feature_pipeline.named_steps[
        "preprocesador"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    classifier = fitted_pipeline.named_steps[
        "classifier"
    ]

    feature_importances = classifier.feature_importances_

    if len(feature_names) != len(feature_importances):
        raise ValueError(
            "La cantidad de nombres de features no coincide "
            "con la cantidad de importancias del modelo."
        )

    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importances,
        }
    )

    return (
        importance_table
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# Grafica las 20 features transformadas con mayor importancia
def create_feature_importance_figure(
    importance_table: pd.DataFrame,
    top_n: int = 20,
):

    top_features = (
        importance_table
        .head(top_n)
        .sort_values(
            by="importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.barh(
        top_features["feature"],
        top_features["importance"],
        color="steelblue",
    )

    axis.set_title(
        f"Top {top_n} features más importantes"
    )
    axis.set_xlabel("Importancia")
    axis.set_ylabel("Feature transformada")

    figure.tight_layout()

    return figure


# Parte 5: Registro de la evaluación final en MLflow

# Entrena, evalúa y registra el modelo seleccionado
# Solo se marcará como candidato al Model Registry si supera la validación

def run_final_evaluation(
    X_model_train: pd.DataFrame,
    X_model_test: pd.DataFrame, # variables predictores de v2
    X_audit_test: pd.DataFrame, # variables sensibles solo para auditoría
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:

    tags = build_mlflow_tags()
    tags.update(
        {
            "evaluation_stage": "final_test",
            "selected_tuning_run_id": (
                SELECTED_TUNING_RUN_ID
            ),
            "evaluation_code_sha256": calculate_file_hash(
                Path(__file__).resolve()
            ),
        }
    )

    parameters = {
        "algorithm": SELECTED_ALGORITHM,
        "feature_set": SELECTED_FEATURE_SET,
        "random_seed": RANDOM_SEED,
        "data_version": DATA_VERSION,
        "test_size": TEST_SIZE,
        "classifier__n_estimators": 150,
        "classifier__max_depth": 20,
        "classifier__min_samples_split": 5,
        "classifier__min_samples_leaf": 2,
        "classifier__max_features": "sqrt",
        "classifier__class_weight": "balanced",
        "classifier__n_jobs": 1,
    }

    with mlflow.start_run(
        run_name="final-random-forest-evaluation",
        tags=tags,
    ) as active_run:

        mlflow.log_params(parameters)

        mlflow.log_dict(
            {
                "selected_algorithm": SELECTED_ALGORITHM,
                "selected_feature_set": SELECTED_FEATURE_SET,
                "selected_tuning_run_id": (
                    SELECTED_TUNING_RUN_ID
                ),
                "selection_metric": "cv_g_mean_mean",
                "validation_g_mean": VALIDATION_G_MEAN,
                "selection_reason": (
                    "Highest cross-validation G-Mean, F1 and "
                    "ROC AUC among tuned candidates."
                ),
            },
            "configuration/selection.json",
        )

        mlflow.log_dict(
            {
                "minimum_test_g_mean": MINIMUM_TEST_G_MEAN,
                "maximum_g_mean_difference": (
                    MAXIMUM_G_MEAN_DIFFERENCE
                ),
                "minimum_test_recall": MINIMUM_TEST_RECALL,
                "minimum_test_specificity": (
                    MINIMUM_TEST_SPECIFICITY
                ),
            },
            "configuration/validation_criteria.json",
        )

        fitted_pipeline = build_selected_pipeline()

        fitted_pipeline.fit(
            X_model_train,
            y_train,
        )

        evaluation_summary, predicted_classes, positive_probabilities = (
            evaluate_final_pipeline(
                fitted_pipeline,
                X_model_test,
                X_audit_test,
                y_test,
            )
        )

        validation_passed, validation_checks = (
            validate_final_model(
                evaluation_summary
            )
        )

        validation_status = (
            "validation_passed"
            if validation_passed
            else "validation_failed"
        )

        mlflow.set_tags(
            {
                "validation_status": validation_status,
                "registry_candidate": str(
                    validation_passed
                ),
            }
        )

        mlflow.log_metrics(
            build_mlflow_metrics(
                evaluation_summary
            )
        )

        mlflow.log_dict(
            {
                "validation_passed": validation_passed,
                "validation_status": validation_status,
                "checks": validation_checks,
                "metrics": evaluation_summary,
            },
            "validation/final_validation.json",
        )

        importance_table = (
            create_feature_importance_table(
                fitted_pipeline
            )
        )

        mlflow.log_table(
            data=importance_table,
            artifact_file=(
                "tables/feature_importance.json"
            ),
        )

        importance_figure = (
            create_feature_importance_figure(
                importance_table,
                top_n=20,
            )
        )

        try:
            mlflow.log_figure(
                importance_figure,
                "plots/feature_importance_top_20.png",
            )
        finally:
            plt.close(importance_figure)

        confusion_figure = create_confusion_matrix_figure(
            y_test,
            predicted_classes,
            "final-random-forest-evaluation",
            evaluation_label="Test",
        )

        try:
            mlflow.log_figure(
                confusion_figure,
                "plots/confusion_matrix_test.png",
            )
        finally:
            plt.close(confusion_figure)

        roc_figure = create_roc_curve_figure(
            y_test,
            positive_probabilities,
            "final-random-forest-evaluation",
            evaluation_label="Test",
        )

        try:
            mlflow.log_figure(
                roc_figure,
                "plots/roc_curve_test.png",
            )
        finally:
            plt.close(roc_figure)

        input_example = (
            X_model_train
            .head(5)
            .copy()
        )

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

        model_info = mlflow.sklearn.log_model(
            sk_model=fitted_pipeline,
            name="model",
            signature=model_signature,
            input_example=input_example,
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        print(
            f"Estado de validación: "
            f"{validation_status}"
        )
        print(
            f"Run final: "
            f"{active_run.info.run_id}"
        )

        return {
            "run_id": active_run.info.run_id,
            "model_uri": model_info.model_uri,
            "validation_passed": validation_passed,
            "validation_status": validation_status,
            "validation_checks": validation_checks,
            **evaluation_summary,
        }


# Parte 6: Ejecución principal

def main() -> None:

    print("=" * 70)
    print("ACI94 - Evaluación final del modelo seleccionado")
    print("=" * 70)

    configure_mlflow()

    df = load_processed_data()

    X_train, X_test, y_train, y_test = split_data(
        df
    )

    # El modelo recibe únicamente las 11 variables de v2
    X_model_train = select_model_input(
        X_train,
        SELECTED_FEATURE_SET,
    )

    X_model_test = select_model_input(
        X_test,
        SELECTED_FEATURE_SET,
    )

    print(
        f"Entrenamiento del modelo: "
        f"{X_model_train.shape}"
    )
    print(
        f"Evaluación final: "
        f"{X_model_test.shape}"
    )

    final_result = run_final_evaluation(
        X_model_train=X_model_train,
        X_model_test=X_model_test,
        X_audit_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("\nResultados finales:")
    print(
        f"G-Mean: "
        f"{final_result['test_g_mean']:.4f}"
    )
    print(
        f"F1: "
        f"{final_result['test_f1']:.4f}"
    )
    print(
        f"Recall: "
        f"{final_result['test_recall']:.4f}"
    )
    print(
        f"Especificidad: "
        f"{final_result['test_specificity']:.4f}"
    )
    print(
        f"ROC AUC: "
        f"{final_result['test_roc_auc']:.4f}"
    )
    print(
        f"Diferencia absoluta validación-test: "
        f"{final_result['validation_test_g_mean_absolute_difference']:.4f}"
    )

    print("\nResultados por subgrupo:")
    print(
        f"G-Mean Female: "
        f"{final_result['test_subgroup_female_g_mean']:.4f}"
    )
    print(
        f"G-Mean Male: "
        f"{final_result['test_subgroup_male_g_mean']:.4f}"
    )
    print(
        f"Gap de G-Mean: "
        f"{final_result['test_subgroup_gap_g_mean']:.4f}"
    )
    print(
        f"Gap de recall: "
        f"{final_result['test_subgroup_gap_recall']:.4f}"
    )
    print(
        f"Gap de especificidad: "
        f"{final_result['test_subgroup_gap_specificity']:.4f}"
    )

    print(
        f"\nEstado final: "
        f"{final_result['validation_status']}"
    )


if __name__ == "__main__":
    main()