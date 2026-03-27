"""
test_api.py
Automated tests for the FastAPI classifier backend.

Run:
    pytest test_api.py -v

Requires:
    pip install httpx pytest
    python train_model.py   (to build model.joblib & model_priority.joblib)
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# ── Health check ─────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self):
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ── /predict ─────────────────────────────────────────────────

class TestPredict:
    def test_technical_support(self):
        resp = client.post("/predict", json={"text": "my internet keeps disconnecting"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "Technical Support"
        assert data["priority"] in ("High", "Medium", "Low")
        assert 0.0 <= data["confidence"] <= 1.0
        assert "routing_reason" in data
        assert data["model"] == "tfidf-sgd"

    def test_billing(self):
        resp = client.post("/predict", json={"text": "I was charged twice yesterday"})
        assert resp.status_code == 200
        assert resp.json()["category"] == "Billing"

    def test_account_management(self):
        resp = client.post("/predict", json={"text": "I forgot my account password"})
        assert resp.status_code == 200
        assert resp.json()["category"] == "Account Management"

    def test_blank_text_raises_422(self):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_too_short_text_raises_422(self):
        resp = client.post("/predict", json={"text": "ab"})
        assert resp.status_code == 422

    def test_missing_text_field_raises_422(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_response_has_all_fields(self):
        resp = client.post("/predict", json={"text": "network is down in my area"})
        data = resp.json()
        for field in ("category", "priority", "confidence", "routing_reason", "model"):
            assert field in data, f"Missing field: {field}"

    def test_confidence_is_float_between_0_and_1(self):
        resp = client.post("/predict", json={"text": "please upgrade my plan"})
        assert resp.status_code == 200
        conf = resp.json()["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0


# ── /predict/batch ───────────────────────────────────────────

class TestPredictBatch:
    def test_batch_two_tickets(self):
        resp = client.post(
            "/predict/batch",
            json={"tickets": ["my bill is wrong", "cannot connect to wifi"]},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2

    def test_batch_order_preserved(self):
        tickets = [
            "I was charged twice",
            "my internet keeps disconnecting",
            "forgot my password",
        ]
        resp = client.post("/predict/batch", json={"tickets": tickets})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 3
        assert results[0]["category"] == "Billing"
        assert results[1]["category"] == "Technical Support"
        assert results[2]["category"] == "Account Management"

    def test_batch_blank_entry_raises_422(self):
        resp = client.post(
            "/predict/batch",
            json={"tickets": ["valid ticket", "   "]},
        )
        assert resp.status_code == 422

    def test_batch_empty_list_raises_422(self):
        resp = client.post("/predict/batch", json={"tickets": []})
        assert resp.status_code == 422

    def test_batch_single_ticket(self):
        resp = client.post("/predict/batch", json={"tickets": ["network outage in sector 4"]})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1


# ── OpenAPI schema ───────────────────────────────────────────

class TestOpenAPI:
    def test_openapi_json_accessible(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_docs_accessible(self):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_accessible(self):
        resp = client.get("/redoc")
        assert resp.status_code == 200
