from pathlib import Path

import mlflow
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "production"

MODEL_NAME = "adult-income-classifier"
MODEL_VERSION = "1"


class ModelService:
    def __init__(self):
        self.model = mlflow.sklearn.load_model(str(MODEL_PATH))

    def predict(self, input_data: dict) -> dict:
        dataframe = pd.DataFrame([input_data])

        prediction = self.model.predict(dataframe)[0]
        probabilities = self.model.predict_proba(dataframe)[0]

        classes = list(self.model.classes_)
        positive_class = ">50K"

        positive_index = classes.index(positive_class)
        positive_probability = float(probabilities[positive_index])

        return {
            "prediction": str(prediction),
            "probability": positive_probability,
            "model_version": MODEL_VERSION,
        }


model_service = ModelService()
