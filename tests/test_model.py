from pathlib import Path

import mlflow
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "production"


def test_model_loads_and_predicts():
    model = mlflow.sklearn.load_model(str(MODEL_PATH))

    input_data = pd.DataFrame(
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

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    assert prediction in ["<=50K", ">50K"]
    assert len(probabilities) == 2
    assert all(0 <= probability <= 1 for probability in probabilities)
    assert abs(sum(probabilities) - 1.0) < 1e-6