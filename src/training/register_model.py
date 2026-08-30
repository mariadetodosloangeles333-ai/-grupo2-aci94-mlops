"""
Registro, validación y promoción del modelo seleccionado mediante MLflow.

Este archivo toma únicamente el modelo aprobado en la evaluación final,
lo registra como Candidate, comprueba que pueda cargarse y generar
predicciones válidas, y lo promueve a Production si supera la validación.
"""


# Parte 1: Importaciones y configuración general

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient


# Ubicar la carpeta principal del repositorio
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Permitir ejecutar register_model.py directamente
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar el nombre registrado y el feature set seleccionados
from src.training.evaluate import (  # noqa: E402
    REGISTERED_MODEL_NAME,
    SELECTED_FEATURE_SET,
)

# Importar las constantes y funciones reutilizadas del entrenamiento
from src.training.train import (  # noqa: E402
    EXPECTED_CLASSES,
    TARGET_COLUMN,
    configure_mlflow,
    load_processed_data,
    select_model_input,
)


# Run final que superó los criterios de validación
FINAL_RUN_ID = "11c8bc44969d40938763b78ead476dfd"

# Ubicación del modelo dentro del run final
MODEL_ARTIFACT_PATH = "model"

# Alias que representan el ciclo del modelo
CANDIDATE_ALIAS = "candidate"
VALIDATION_ALIAS = "validation"
PRODUCTION_ALIAS = "production"

# Parte 2: Validación del run de origen

# Comprueba que el run final terminó correctamente y fue aprobado
def validate_source_run(
    client: MlflowClient,
):

    source_run = client.get_run(
        FINAL_RUN_ID
    )

    if source_run.info.status != "FINISHED":
        raise ValueError(
            "El run final no terminó correctamente y no puede registrarse."
        )

    validation_status = source_run.data.tags.get(
        "validation_status"
    )

    registry_candidate = source_run.data.tags.get(
        "registry_candidate"
    )

    if validation_status != "validation_passed":
        raise ValueError(
            "El run final no superó los criterios de validación."
        )

    if registry_candidate != "True":
        raise ValueError(
            "El run final no está marcado como candidato al Registry."
        )

    required_metrics = [
        "test_g_mean",
        "test_f1",
        "test_recall",
        "test_specificity",
        "test_roc_auc",
    ]

    missing_metrics = [
        metric_name
        for metric_name in required_metrics
        if metric_name not in source_run.data.metrics
    ]

    if missing_metrics:
        raise ValueError(
            "El run final no contiene todas las métricas requeridas: "
            f"{missing_metrics}"
        )

    print(
        "Run de origen validado correctamente: "
        f"{FINAL_RUN_ID}"
    )

    return source_run


# Parte 3: Registro del modelo candidato

# Busca si el run final ya fue registrado como una versión del modelo
def find_existing_model_version(
    client: MlflowClient,
):

    try:
        model_versions = client.search_model_versions(
            filter_string=(
                f"name = '{REGISTERED_MODEL_NAME}'"
            )
        )
    except mlflow.exceptions.MlflowException:
        return None

    for model_version in model_versions:
        if model_version.run_id == FINAL_RUN_ID:
            return model_version

    return None


# Registra el modelo aprobado y lo identifica como Candidate
def register_candidate_model(
    client: MlflowClient,
):

    existing_version = find_existing_model_version(
        client
    )

    if existing_version is not None:
        registered_version = existing_version
        print(
            "El modelo ya estaba registrado como versión "
            f"{registered_version.version}."
        )
    else:
        model_uri = (
            f"runs:/{FINAL_RUN_ID}/"
            f"{MODEL_ARTIFACT_PATH}"
        )

        registered_version = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )

        print(
            "Nueva versión registrada: "
            f"{registered_version.version}"
        )

    version = str(
        registered_version.version
    )

    # Registrar el estado inicial dentro del ciclo del modelo
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="lifecycle_status",
        value="Candidate",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="registry_validation_status",
        value="pending",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="source_run_id",
        value=FINAL_RUN_ID,
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CANDIDATE_ALIAS,
        version=version,
    )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=(
            "Random Forest entrenado con v2_without_sensitive. "
            "Superó los criterios de evaluación final y quedó "
            "pendiente de validación técnica para producción."
        ),
    )

    print(
        f"Modelo registrado como "
        f"{REGISTERED_MODEL_NAME} versión {version}."
    )
    print(
        f"Alias asignado: @{CANDIDATE_ALIAS}"
    )

    return registered_version


# Parte 4: Validación técnica del modelo registrado

# Comprueba que el Candidate pueda cargarse y generar predicciones válidas
def validate_registered_candidate(
    client: MlflowClient,
):

    candidate_version = (
        client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CANDIDATE_ALIAS,
        )
    )

    version = str(
        candidate_version.version
    )

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}"
        f"@{CANDIDATE_ALIAS}"
    )

    # Cargar el modelo directamente desde Model Registry
    registered_model = mlflow.sklearn.load_model(
        model_uri
    )

    # Construir una muestra con la misma estructura de producción
    df = load_processed_data()

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    validation_sample = (
        select_model_input(
            X,
            SELECTED_FEATURE_SET,
        )
        .head(10)
        .copy()
    )

    predictions = registered_model.predict(
        validation_sample
    )

    probability_matrix = registered_model.predict_proba(
        validation_sample
    )

    validation_checks = {
        "prediction_count": (
            len(predictions)
            == len(validation_sample)
        ),
        "expected_classes": (
            set(predictions).issubset(
                EXPECTED_CLASSES
            )
        ),
        "probability_shape": (
            probability_matrix.shape
            == (len(validation_sample), 2)
        ),
        "finite_probabilities": bool(
            np.isfinite(
                probability_matrix
            ).all()
        ),
        "valid_probability_range": bool(
            (
                (probability_matrix >= 0.0)
                & (probability_matrix <= 1.0)
            ).all()
        ),
        "probability_sum": bool(
            np.allclose(
                probability_matrix.sum(axis=1),
                1.0,
            )
        ),
    }

    validation_passed = all(
        validation_checks.values()
    )

    validation_status = (
        "passed"
        if validation_passed
        else "failed"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="registry_validation_status",
        value=validation_status,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="smoke_test_checks",
        value=str(validation_checks),
    )

    if not validation_passed:
        raise ValueError(
            "El modelo registrado no superó el smoke test: "
            f"{validation_checks}"
        )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="lifecycle_status",
        value="Validation",
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=VALIDATION_ALIAS,
        version=version,
    )

    print(
        "Smoke test del modelo registrado: PASS"
    )
    print(
        f"Comprobaciones: {validation_checks}"
    )
    print(
        f"Alias asignado: @{VALIDATION_ALIAS}"
    )

    return {
        "version": version,
        "validation_passed": validation_passed,
        "validation_checks": validation_checks,
        "predictions": predictions.tolist(),
    }


# Parte 5: Promoción del modelo validado

# Promueve a Production únicamente la versión que superó el smoke test
def promote_validated_model(
    client: MlflowClient,
):

    validated_version = (
        client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=VALIDATION_ALIAS,
        )
    )

    version = str(
        validated_version.version
    )

    validation_status = (
        validated_version.tags.get(
            "registry_validation_status"
        )
    )

    if validation_status != "passed":
        raise ValueError(
            "La versión no superó la validación técnica "
            "y no puede promoverse a Production."
        )

    # El alias Production identificará la versión utilizada en inferencia
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=version,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="lifecycle_status",
        value="Production",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="candidate_status",
        value="approved",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="deployment_status",
        value="production",
    )

    client.set_registered_model_tag(
        name=REGISTERED_MODEL_NAME,
        key="problem_type",
        value="binary_classification",
    )

    client.set_registered_model_tag(
        name=REGISTERED_MODEL_NAME,
        key="production_version",
        value=version,
    )

    client.set_registered_model_tag(
        name=REGISTERED_MODEL_NAME,
        key="feature_set",
        value=SELECTED_FEATURE_SET,
    )

    # Los estados anteriores quedan documentados mediante tags.
    # Los alias temporales se retiran para reflejar el estado actual.
    client.delete_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CANDIDATE_ALIAS,
    )

    client.delete_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=VALIDATION_ALIAS,
    )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=(
            "Modelo Random Forest seleccionado mediante validación "
            "cruzada y búsqueda de hiperparámetros. Utiliza "
            "v2_without_sensitive, superó la evaluación final y "
            "el smoke test del Model Registry. Las variables "
            "sensibles se conservan únicamente para auditoría."
        ),
    )

    print(
        f"Modelo promovido a Production: "
        f"{REGISTERED_MODEL_NAME} versión {version}"
    )
    print(
        f"URI de producción: "
        f"models:/{REGISTERED_MODEL_NAME}"
        f"@{PRODUCTION_ALIAS}"
    )

    return {
        "model_name": REGISTERED_MODEL_NAME,
        "version": version,
        "alias": PRODUCTION_ALIAS,
        "model_uri": (
            f"models:/{REGISTERED_MODEL_NAME}"
            f"@{PRODUCTION_ALIAS}"
        ),
    }


# Parte 6: Ejecución principal

def main() -> None:

    print("=" * 70)
    print("ACI94 - Registro y promoción del modelo")
    print("=" * 70)

    configure_mlflow()

    client = MlflowClient()

    validate_source_run(
        client
    )

    register_candidate_model(
        client
    )

    validate_registered_candidate(
        client
    )

    production_result = promote_validated_model(
        client
    )

    print("=" * 70)
    print("Ciclo completado correctamente:")
    print("Experiment -> Candidate -> Validation -> Production")
    print(
        f"Modelo: {production_result['model_name']}"
    )
    print(
        f"Versión: {production_result['version']}"
    )
    print(
        f"URI: {production_result['model_uri']}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()