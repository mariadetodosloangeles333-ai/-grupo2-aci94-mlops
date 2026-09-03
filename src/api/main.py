import time
from threading import Lock

from fastapi import FastAPI, Request

from src.api.model_service import (
    MODEL_NAME,
    MODEL_VERSION,
    model_service,
)
from src.api.schemas import PredictionRequest, PredictionResponse

# Métricas operativas acumuladas durante la vida del proceso.
# Se reinician cada vez que el servicio o contenedor vuelve a iniciarse.
SERVICE_START_TIME = time.perf_counter()
SYSTEM_METRICS_LOCK = Lock()
SYSTEM_METRICS = {
    "total_requests": 0,
    "error_requests": 0,
    "total_latency_seconds": 0.0,
    "last_latency_seconds": 0.0,
}

app = FastAPI(
    title="Adult Income Classification API",
    description=(
        "API para servir el modelo de clasificación de ingresos "
        "Adult Census Income 1994."
    ),
    version="1.0.0",
)


@app.middleware("http")
async def collect_system_metrics(request: Request, call_next):
    """
    Registra volumen, errores y latencia de las solicitudes HTTP.

    El endpoint /metrics se excluye para evitar que consultar las métricas
    modifique los mismos contadores que se están observando.
    """
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_seconds = time.perf_counter() - start_time

        with SYSTEM_METRICS_LOCK:
            SYSTEM_METRICS["total_requests"] += 1
            SYSTEM_METRICS["total_latency_seconds"] += latency_seconds
            SYSTEM_METRICS["last_latency_seconds"] = latency_seconds

            if status_code >= 400:
                SYSTEM_METRICS["error_requests"] += 1


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "adult-income-api",
    }


@app.get("/metrics")
def system_metrics():
    """
    Expone métricas operativas acumuladas durante la vida del servicio.

    La disponibilidad histórica requeriría una herramienta externa; este
    endpoint informa el estado actual y el tiempo activo del proceso.
    """
    uptime_seconds = time.perf_counter() - SERVICE_START_TIME

    with SYSTEM_METRICS_LOCK:
        total_requests = SYSTEM_METRICS["total_requests"]
        error_requests = SYSTEM_METRICS["error_requests"]
        total_latency_seconds = SYSTEM_METRICS["total_latency_seconds"]
        last_latency_seconds = SYSTEM_METRICS["last_latency_seconds"]

    successful_requests = total_requests - error_requests
    error_rate = (
        error_requests / total_requests
        if total_requests > 0
        else 0.0
    )
    average_latency_seconds = (
        total_latency_seconds / total_requests
        if total_requests > 0
        else 0.0
    )
    throughput = (
        total_requests / uptime_seconds
        if uptime_seconds > 0
        else 0.0
    )

    return {
        "availability_status": "up",
        "uptime_seconds": round(uptime_seconds, 3),
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "error_requests": error_requests,
        "error_rate": round(error_rate, 6),
        "throughput_requests_per_second": round(throughput, 6),
        "average_latency_ms": round(average_latency_seconds * 1000, 3),
        "last_latency_ms": round(last_latency_seconds * 1000, 3),
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
