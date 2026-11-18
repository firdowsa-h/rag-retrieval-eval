from fastapi.testclient import TestClient
from app.main import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rejects_invalid_strategy():
    response = TestClient(app).get("/search", params={"q": "indexes", "strategy": "unknown"})
    assert response.status_code == 422


def test_rejects_bad_k():
    response = TestClient(app).get("/search", params={"q": "indexes", "k": 0})
    assert response.status_code == 422
