"""
Búsqueda reproducible de hiperparámetros para los modelos candidatos definidos 
en la comparación inicial

Se utiliza únicamente el conjunto de entrenamiento, validación cruzada
estratificada y el feature set sin variables sensibles. Todos los resultados
relevantes se registraran en MLflow.
"""


# Parte 1: Importaciones y configuración general

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)

# Ubicar la carpeta principal del repositorio
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Permitir ejecutar tune.py directamente
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importación de feature set sin variables sensibles de feature engineering
from src.features.build_features import (  # noqa: E402
    FEATURE_SET_WITHOUT_SENSITIVE_VERSION,
)

# Importación de todo lo necesario desde train.py
from src.training.train import (  # noqa: E402
    CV_FOLDS,
    DATA_VERSION,
    RANDOM_SEED,
    TEST_SIZE,
    build_mlflow_metrics,
    build_mlflow_tags,
    build_model_pipeline,
    calculate_classification_metrics,
    calculate_file_hash,
    calculate_subgroup_metrics,
    configure_mlflow,
    create_confusion_matrix_figure,
    create_roc_curve_figure,
    create_stratified_cross_validation,
    generate_out_of_fold_predictions,
    get_cross_validation_scoring,
    load_processed_data,
    select_model_input,
    split_data,
    summarize_subgroup_metrics,
)

# Feature set seleccionado después de la comparación inicial
SELECTED_FEATURE_SET = FEATURE_SET_WITHOUT_SENSITIVE_VERSION

# Métrica principal para seleccionar hiperparámetros
PRIMARY_METRIC = "g_mean"


# Parte 2: Modelos candidatos y espacios de búsqueda

# Define los hiperparámetros que se evaluarán para cada candidato
def get_tuning_experiments() -> list[dict]:

    return [
        {
            # Se utiliza Grid Search porque el espacio de búsqueda es pequeño.
            "run_name": "tuning-logistic-regression-balanced",
            "algorithm": "LogisticRegression",
            "search_type": "grid",
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1_000,
                random_state=RANDOM_SEED,
                solver="liblinear",
            ),
            "parameter_space": {
                "classifier__C": [
                    0.1,
                    1.0,
                    10.0,
                ],
                # 0.0 representa regularización L2 y 1.0 representa L1.
                "classifier__l1_ratio": [
                    0.0,
                    1.0,
                ],
            },
        },
        # Se utiliza Grid Search para evaluar todas las combinaciones definidas.
        {
            "run_name": "tuning-decision-tree-balanced",
            "algorithm": "DecisionTreeClassifier",
            "search_type": "grid",
            "model": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            "parameter_space": {
                "classifier__criterion": [
                    "gini",
                    "entropy",
                ],
                "classifier__max_depth": [
                    5,
                    10,
                    15,
                ],
                "classifier__min_samples_leaf": [
                    10,
                    20,
                    50,
                ],
            },
        },
        # Se utiliza Randomized Search porque el espacio completo contiene
        # demasiadas combinaciones y no se evaluarán todas.
        {
            "run_name": "tuning-random-forest-balanced",
            "algorithm": "RandomForestClassifier",
            "search_type": "random",
            "n_iter": 10,
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            "parameter_space": {
                "classifier__n_estimators": [
                    150,
                    200,
                    300,
                ],
                "classifier__max_depth": [
                    10,
                    15,
                    20,
                    None,
                ],
                "classifier__min_samples_split": [
                    2,
                    5,
                    10,
                ],
                "classifier__min_samples_leaf": [
                    2,
                    5,
                    10,
                ],
                "classifier__max_features": [
                    "sqrt",
                    0.5,
                ],
            },
        },
    ]


# Parte 3: Construcción de las búsquedas de hiperparámetros

# Construye GridSearchCV o RandomizedSearchCV según el experimento
def build_hyperparameter_search(
    experiment: dict,
):

    pipeline = build_model_pipeline(
        classifier=experiment["model"],
        feature_set_version=SELECTED_FEATURE_SET,
    )

    common_arguments = {
        "estimator": pipeline,
        "scoring": get_cross_validation_scoring(),
        "refit": PRIMARY_METRIC,
        "cv": create_stratified_cross_validation(),
        # Paralelizar las combinaciones y folds desde el objeto de búsqueda.
        # Random Forest conserva n_jobs=1 para evitar paralelismo anidado.
        "n_jobs": -1,
        "return_train_score": True,
        "verbose": 1,
        "error_score": "raise",
    }

    # Uso del grid searcg
    if experiment["search_type"] == "grid":
        return GridSearchCV(
            param_grid=experiment["parameter_space"],
            **common_arguments,
        )

    # Uso de randomized search
    if experiment["search_type"] == "random":
        return RandomizedSearchCV(
            param_distributions=(
                experiment["parameter_space"]
            ),
            n_iter=experiment["n_iter"],
            random_state=RANDOM_SEED,
            **common_arguments,
        )
    
    # Validación
    raise ValueError(
        "Tipo de búsqueda no reconocido: "
        f"{experiment['search_type']}"
    )


# Parte 4: Organización de los resultados de búsqueda

# Convierte los resultados de una búsqueda en una tabla comparable
def create_search_results_table(
    search,
    experiment: dict,
) -> pd.DataFrame:

    raw_results = pd.DataFrame(
        search.cv_results_
    )

    # Hiperparámetros utilizados en cada combinación
    parameter_results = pd.json_normalize(
        raw_results["params"]
    )

    # Resultados principales de entrenamiento y validación
    metric_results = pd.DataFrame(
        {
            "algorithm": experiment["algorithm"],
            "search_type": experiment["search_type"],
            "candidate_index": raw_results.index,
            "rank_g_mean": raw_results[
                "rank_test_g_mean"
            ],
            "train_g_mean_mean": raw_results[
                "mean_train_g_mean"
            ],
            "cv_g_mean_mean": raw_results[
                "mean_test_g_mean"
            ],
            "cv_g_mean_std": raw_results[
                "std_test_g_mean"
            ],
            "cv_f1_mean": raw_results[
                "mean_test_f1"
            ],
            "cv_recall_mean": raw_results[
                "mean_test_recall"
            ],
            "cv_specificity_mean": raw_results[
                "mean_test_specificity"
            ],
            "cv_roc_auc_mean": raw_results[
                "mean_test_roc_auc"
            ],
            "fit_time_mean": raw_results[
                "mean_fit_time"
            ],
        }
    )

    # Diferencia utilizada para revisar sobreajuste
    metric_results["gap_g_mean"] = (
        metric_results["train_g_mean_mean"]
        - metric_results["cv_g_mean_mean"]
    )

    # Tabla de resultados
    results_table = pd.concat(
        [
            metric_results,
            parameter_results,
        ],
        axis=1,
    )

    return (
        results_table
        .sort_values(
            by="rank_g_mean",
            ascending=True,
        )
        .reset_index(drop=True)
    )


# Parte 5: Evaluación out-of-fold del mejor pipeline

# Evalúa el mejor pipeline con predicciones out-of-fold y las mismas
# métricas utilizadas en la comparación inicial. El test permanece reservado
def evaluate_best_pipeline(
    search,
    X_model_train: pd.DataFrame,
    X_audit_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    dict[str, float],
    object,
    object,
]:

    best_pipeline = search.best_estimator_

    predicted_classes, positive_probabilities = (
        generate_out_of_fold_predictions(
            best_pipeline,
            X_model_train,
            y_train,
        )
    )

    general_metrics = calculate_classification_metrics(
        y_train,
        predicted_classes,
        positive_probabilities,
    )

    general_summary = {
        f"oof_{metric_name}": float(metric_value)
        for metric_name, metric_value
        in general_metrics.items()
    }

    subgroup_results = calculate_subgroup_metrics(
        X_audit_train,
        y_train,
        predicted_classes,
        positive_probabilities,
    )

    subgroup_summary = summarize_subgroup_metrics(
        subgroup_results
    )

    evaluation_summary = {
        **general_summary,
        **subgroup_summary,
    }

    return (
        evaluation_summary,
        predicted_classes,
        positive_probabilities,
    )


# Parte 6: Ejecución y registro de una búsqueda en MLflow

# Ejecuta una búsqueda completa y registra su mejor resultado
def run_tuning_experiment(
    experiment: dict,
    X_model_train: pd.DataFrame,
    X_audit_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:

    tags = build_mlflow_tags()
    tags.update(
        {
            "evaluation_stage": "hyperparameter_tuning",
            "tuning_code_sha256": calculate_file_hash(
                Path(__file__).resolve()
            ),
        }
    )

    general_parameters = {
        "algorithm": experiment["algorithm"],
        "search_type": experiment["search_type"],
        "feature_set": SELECTED_FEATURE_SET,
        "primary_metric": PRIMARY_METRIC,
        "random_seed": RANDOM_SEED,
        "data_version": DATA_VERSION,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
    }

    if experiment["search_type"] == "random":
        general_parameters["n_iter"] = experiment["n_iter"]

    with mlflow.start_run(
        run_name=experiment["run_name"],
        tags=tags,
    ) as active_run:

        mlflow.log_params(general_parameters)

        # Registrar el espacio que será explorado
        mlflow.log_dict(
            experiment["parameter_space"],
            "configuration/parameter_space.json",
        )

        search = build_hyperparameter_search(
            experiment
        )

        print(
            f"\nIniciando búsqueda: "
            f"{experiment['run_name']}"
        )

        search.fit(
            X_model_train,
            y_train,
        )

        results_table = create_search_results_table(
            search,
            experiment,
        )

        evaluation_summary, predicted_classes, positive_probabilities = (
            evaluate_best_pipeline(
                search,
                X_model_train,
                X_audit_train,
                y_train,
            )
        )

        best_parameters = {
            f"best_{parameter_name}": parameter_value
            for parameter_name, parameter_value
            in search.best_params_.items()
        }

        mlflow.log_params(best_parameters)

        best_row = results_table.iloc[0].to_dict()

        search_metric_names = [
            "train_g_mean_mean",
            "cv_g_mean_mean",
            "cv_g_mean_std",
            "cv_f1_mean",
            "cv_recall_mean",
            "cv_specificity_mean",
            "cv_roc_auc_mean",
            "fit_time_mean",
            "gap_g_mean",
        ]

        best_search_metrics = {
            f"best_{metric_name}": float(
                best_row[metric_name]
            )
            for metric_name in search_metric_names
        }

        all_metrics = {
            **best_search_metrics,
            **evaluation_summary,
        }

        mlflow.log_metrics(
            build_mlflow_metrics(all_metrics)
        )

        # Sustituir NaN por None para guardar una tabla JSON válida
        results_for_logging = (
            results_table
            .astype(object)
            .where(
                pd.notna(results_table),
                None,
            )
        )

        # Conservar todas las combinaciones permite auditar por qué una
        # configuración fue seleccionada sobre las demás.
        mlflow.log_table(
            data=results_for_logging,
            artifact_file="tables/search_results.json",
        )

        mlflow.log_dict(
            {
                "best_parameters": search.best_params_,
                "best_cv_g_mean": float(
                    search.best_score_
                ),
                "number_of_candidates": int(
                    len(results_table)
                ),
                "feature_set": SELECTED_FEATURE_SET,
                "primary_metric": PRIMARY_METRIC,
            },
            "configuration/best_result.json",
        )

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

        best_pipeline = search.best_estimator_

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

        example_predictions = best_pipeline.predict(
            input_example
        )

        model_signature = infer_signature(
            input_example,
            example_predictions,
        )

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            signature=model_signature,
            input_example=input_example,
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        print(
            f"Mejor G-Mean CV: "
            f"{search.best_score_:.4f}"
        )
        print(
            f"Mejores parámetros: "
            f"{search.best_params_}"
        )
        print(
            f"Run registrado: "
            f"{active_run.info.run_id}"
        )

        return {
            "run_name": experiment["run_name"],
            "algorithm": experiment["algorithm"],
            "feature_set": SELECTED_FEATURE_SET,
            **best_search_metrics,
            **evaluation_summary,
        }


# Parte 7: Organización de los mejores resultados

# Construye una tabla con el ganador de cada familia de modelos
def create_tuning_summary_table(
    tuning_results: list[dict],
) -> pd.DataFrame:

    if not tuning_results:
        raise ValueError(
            "No existen resultados de tuning para comparar."
        )

    results_table = pd.DataFrame(
        tuning_results
    )

    return (
        results_table
        .sort_values(
            by="best_cv_g_mean_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# Parte 8: Ejecución principal

def main() -> None:

    print("=" * 70)
    print("ACI94 - Búsqueda de hiperparámetros")
    print("=" * 70)

    configure_mlflow()

    df = load_processed_data()

    X_train, X_test, y_train, y_test = split_data(
        df
    )

    # El modelado utiliza únicamente las variables de v2
    X_model_train = select_model_input(
        X_train,
        SELECTED_FEATURE_SET,
    )

    print(
        f"Variables para modelado: "
        f"{X_model_train.shape[1]}"
    )
    print(
        f"Variables conservadas para auditoría: "
        f"{X_train.shape[1]}"
    )

    tuning_results = []

    for experiment in get_tuning_experiments():
        result = run_tuning_experiment(
            experiment=experiment,
            X_model_train=X_model_train,
            X_audit_train=X_train,
            y_train=y_train,
        )

        tuning_results.append(result)

    summary_table = create_tuning_summary_table(
        tuning_results
    )

    summary_columns = [
        "run_name",
        "algorithm",
        "feature_set",
        "best_train_g_mean_mean",
        "best_cv_g_mean_mean",
        "best_gap_g_mean",
        "best_cv_g_mean_std",
        "best_cv_f1_mean",
        "best_cv_recall_mean",
        "best_cv_specificity_mean",
        "best_cv_roc_auc_mean",
        "subgroup_gap_g_mean",
        "subgroup_gap_recall",
        "subgroup_gap_specificity",
    ]

    print("\nMejores resultados por familia:")
    print(
        summary_table[summary_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nConjunto de test todavía reservado:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")


if __name__ == "__main__":
    main()


"""
Conclusión de esta etapa:

Se realizó una búsqueda de los mejores parametros para cada modelo candidato visto en los
experimentos iniciales. Fue una comparación bastante ajustada pero al final el modelo de
Random Forest fue el modelo seleccionado ya que obtuvo mayor G-Mean, mayor F1, mayor ROC AUC, 
logra el mejor equilibrio general entre recall y especificidad, presenta la menor diferencia 
de G-Mean entre subgrupos y mejora la especificidad sin reducir excesivamente el recall. Aunque 
su gap entre entrenamiento y validación es mayor, aunque todavía no muestra evidencia crítica de 
sobreajuste, por lo que se deberá hacer una correcta revisión con el conjunto de prueba.
"""