"""
Pruebas automatizadas para los componentes de monitoring.

Verifican:
    - Contrato de entrada del modelo.
    - Data Quality Gates.
    - Clasificación de drift mediante PSI.
    - Decisiones de reentrenamiento.
"""

import pandas as pd
import pytest

from src.monitoring.data_quality_gates import validate_batch
from src.monitoring.drift_detection import (
    classify_psi,
    evaluate_drift_for_batch,
)
from src.monitoring.retraining_decision import decide_retraining
from src.monitoring.model_performance import evaluate_labeled_batch


@pytest.fixture
def valid_batch():
    """Crea un batch válido con las nueve variables utilizadas por la API."""
    return pd.DataFrame(
        [
            {
                "age": 35,
                "education-num": 13,
                "hours-per-week": 40,
                "capital-gain": 0,
                "capital-loss": 0,
                "workclass": "Private",
                "marital-status": "Never-married",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
            }
        ]
    )


# ---------------------------------------------------------------------------
# DATA QUALITY GATES
# ---------------------------------------------------------------------------
def test_valid_batch_has_no_incidents(valid_batch):
    """Un batch que cumple el contrato productivo no debe generar incidentes."""
    incidents = validate_batch(valid_batch)

    assert incidents == []


def test_missing_required_column_is_blocked(valid_batch):
    """La ausencia de una variable requerida debe bloquear el batch."""
    batch_without_age = valid_batch.drop(columns=["age"])

    incidents = validate_batch(batch_without_age)

    assert any(
        incident["rule"] == "missing_required_columns"
        and incident["severity"] == "BLOCK"
        for incident in incidents
    )


def test_unexpected_column_is_blocked(valid_batch):
    """Una columna no contemplada en el contrato debe bloquear el batch."""
    batch_with_extra_column = valid_batch.copy()
    batch_with_extra_column["unexpected_field"] = "unexpected_value"

    incidents = validate_batch(batch_with_extra_column)

    assert any(
        incident["rule"] == "unexpected_columns"
        and incident["severity"] == "BLOCK"
        for incident in incidents
    )


def test_unknown_occupation_generates_warning(valid_batch):
    """Una ocupación desconocida debe generar advertencia sin causar KeyError."""
    batch_with_unknown_occupation = valid_batch.copy()
    batch_with_unknown_occupation.loc[0, "occupation"] = "NEW_OCCUPATION"

    incidents = validate_batch(batch_with_unknown_occupation)

    assert any(
        incident["rule"] == "unknown_category_occupation"
        and incident["severity"] == "WARN"
        for incident in incidents
    )


def test_age_outside_valid_range_is_blocked(valid_batch):
    """Una edad fuera del rango productivo de 17 a 90 debe bloquear el batch."""
    batch_with_invalid_age = valid_batch.copy()
    batch_with_invalid_age.loc[0, "age"] = 999

    incidents = validate_batch(batch_with_invalid_age)

    assert any(
        incident["rule"] == "range_check_age"
        and incident["severity"] == "BLOCK"
        for incident in incidents
    )


# ---------------------------------------------------------------------------
# DRIFT DETECTION
# ---------------------------------------------------------------------------

def test_identical_batch_has_no_drift():
    """Un batch idéntico a la referencia debe producir PSI cero y estado OK."""
    reference = pd.DataFrame(
        {
            "age": [20 + (index % 50) for index in range(100)],
            "education-num": [1 + (index % 16) for index in range(100)],
            "hours-per-week": [20 + (index % 40) for index in range(100)],
            "occupation": [
                "Prof-specialty" if index % 2 == 0 else "Sales"
                for index in range(100)
            ],
        }
    )

    result = evaluate_drift_for_batch(
        reference=reference,
        batch=reference.copy(),
    )

    assert result["max_psi"] == pytest.approx(0.0)
    assert result["status"] == "OK"


@pytest.mark.parametrize(
    ("psi_value", "expected_status"),
    [
        (0.00, "OK"),
        (0.09, "OK"),
        (0.10, "WARNING"),
        (0.24, "WARNING"),
        (0.25, "ALERT"),
        (0.50, "ALERT"),
    ],
)
def test_classify_psi_uses_expected_thresholds(psi_value, expected_status):
    """Los valores de PSI deben respetar los umbrales definidos."""
    assert classify_psi(psi_value) == expected_status


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE MONITORING
# ---------------------------------------------------------------------------
def test_labeled_batch_calculates_classification_metrics():
    """Un batch perfectamente clasificado debe producir métricas iguales a uno."""
    y_true = ["<=50K", "<=50K", ">50K", ">50K"]
    y_pred = ["<=50K", "<=50K", ">50K", ">50K"]
    y_probability = [0.10, 0.20, 0.80, 0.90]

    result = evaluate_labeled_batch(
        batch_id="production_batch_perfect",
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    expected_metrics = {
        "precision",
        "recall",
        "specificity",
        "f1",
        "g_mean",
        "roc_auc",
    }

    assert result["batch_id"] == "production_batch_perfect"
    assert result["sample_count"] == 4
    assert result["ground_truth_available"] is True
    assert set(result["metrics"]) == expected_metrics
    assert all(
        metric_value == pytest.approx(1.0)
        for metric_value in result["metrics"].values()
    )
    assert result["performance_degraded"] is False


def test_labeled_batch_detects_performance_degradation():
    """Una caída importante de G-Mean debe activar la señal de deterioro."""
    y_true = pd.Series(["<=50K", "<=50K", ">50K", ">50K"])
    y_pred = ["<=50K", "<=50K", "<=50K", "<=50K"]
    y_probability = [0.10, 0.20, 0.30, 0.40]

    result = evaluate_labeled_batch(
        batch_id="production_batch_degraded",
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    assert result["metrics"]["recall"] == pytest.approx(0.0)
    assert result["metrics"]["g_mean"] == pytest.approx(0.0)
    assert result["g_mean_drop"] == pytest.approx(0.8374)
    assert result["performance_degraded"] is True


def test_labeled_batch_rejects_different_lengths():
    """Etiquetas, predicciones y probabilidades deben tener igual longitud."""
    with pytest.raises(ValueError, match="misma longitud"):
        evaluate_labeled_batch(
            batch_id="invalid_batch",
            y_true=pd.Series(["<=50K", ">50K"]),
            y_pred=["<=50K"],
            y_probability=[0.20, 0.80],
        )


def test_labeled_batch_requires_both_classes():
    """No se deben interpretar G-Mean y AUC si el batch tiene una sola clase."""
    with pytest.raises(ValueError, match="ambas clases"):
        evaluate_labeled_batch(
            batch_id="single_class_batch",
            y_true=["<=50K", "<=50K", "<=50K"],
            y_pred=["<=50K", "<=50K", "<=50K"],
            y_probability=[0.10, 0.20, 0.30],
        )


# ---------------------------------------------------------------------------
# RETRAINING DECISION
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "max_psi",
        "baseline_g_mean",
        "current_g_mean",
        "expected_decision",
    ),
    [
        (0.02, 0.8374, 0.83, "MANTENER"),
        (0.14, 0.8374, 0.81, "REVISAR"),
        (0.02, 0.8374, 0.75, "REVISAR"),
        (0.45, 0.8374, 0.72, "CONSIDERAR REENTRENAMIENTO"),
    ],
)
def test_retraining_decision_combines_drift_and_performance(
    max_psi,
    baseline_g_mean,
    current_g_mean,
    expected_decision,
):
    """
    La decisión debe combinar el drift con la degradación del G-Mean.

    Los valores actuales representan escenarios simulados y no mediciones
    reales de tráfico productivo.
    """
    result = decide_retraining(
        max_psi=max_psi,
        baseline_metric=baseline_g_mean,
        current_metric=current_g_mean,
    )

    assert result.decision == expected_decision