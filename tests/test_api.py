from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_system_metrics_track_requests():
    """Las métricas operativas deben registrar solicitudes y latencia."""
    before = client.get("/metrics").json()

    response = client.get("/health")

    after = client.get("/metrics").json()

    assert response.status_code == 200
    assert after["availability_status"] == "up"
    assert after["total_requests"] == before["total_requests"] + 1
    assert after["successful_requests"] == before["successful_requests"] + 1
    assert after["error_requests"] == before["error_requests"]
    assert after["uptime_seconds"] >= 0
    assert after["throughput_requests_per_second"] >= 0
    assert after["average_latency_ms"] >= 0
    assert after["last_latency_ms"] >= 0


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "adult-income-classifier"
    assert data["model_version"] == "1"
    assert data["model_stage"] == "production"
    assert data["feature_set"] == "v2_without_sensitive"
    assert data["model_loaded"] is True


def test_predict_valid_input():
    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["<=50K", ">50K"]
    assert 0 <= data["probability"] <= 1
    assert data["model_version"] == "1"


def test_predict_rejects_invalid_age():
    payload = {
        "age": 10,
        "education-num": 13,
        "hours-per-week": 40,
        "capital-gain": 0,
        "capital-loss": 0,
        "workclass": "Private",
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
    }

    before = client.get("/metrics").json()

    response = client.post("/predict", json=payload)

    after = client.get("/metrics").json()

    assert response.status_code == 422
    assert after["total_requests"] == before["total_requests"] + 1
    assert after["error_requests"] == before["error_requests"] + 1