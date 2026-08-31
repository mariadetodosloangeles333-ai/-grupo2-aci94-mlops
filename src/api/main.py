from fastapi import FastAPI

from src.api.model_service import (
    MODEL_NAME,
    MODEL_VERSION,
    model_service,
)
from src.api.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="Adult Income Classification API",
    description=(
        "API para servir el modelo de clasificación de ingresos "
        "Adult Census Income 1994."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "adult-income-api",
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_stage": "production",
        "feature_set": "v2_without_sensitive",
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_data = {
        "age": request.age,
        "education-num": request.education_num,
        "hours-per-week": request.hours_per_week,
        "capital-gain": request.capital_gain,
        "capital-loss": request.capital_loss,
        "workclass": request.workclass,
        "marital-status": request.marital_status,
        "occupation": request.occupation,
        "relationship": request.relationship,
    }

    result = model_service.predict(input_data)

    return PredictionResponse(**result)
