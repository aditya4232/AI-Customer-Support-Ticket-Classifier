# FastAPI Backend — Architecture Documentation

## Overview

This document describes the FastAPI REST backend added to the
**AI Customer Support Ticket Classifier** project.

The backend exposes a lightweight, TensorFlow-free HTTP API backed by
a `scikit-learn` TF-IDF + SGD pipeline so that any external client
(mobile app, other microservice, curl, Postman) can classify telecom
support tickets without loading the Streamlit UI.

---

## Directory Layout

```
AI Customer Support Ticket Classifier/
├── train_model.py             ← trains & saves both joblib models
├── model.joblib               ← category classifier (TF-IDF + SGD)
├── model_priority.joblib      ← priority classifier (TF-IDF + SGD)
├── app.py                     ← Streamlit UI (unchanged direct inference)
├── requirements.txt           ← combined deps (Streamlit + FastAPI)
├── test_api.py                ← pytest suite for FastAPI endpoints
├── plans/
│   └── fastapi-backend.md     ← this file
└── backend/
    ├── __init__.py
    ├── main.py                ← FastAPI application
    ├── predictor.py           ← inference logic (singleton model load)
    ├── schemas.py             ← Pydantic v2 request/response models
    └── requirements.txt       ← backend-only deps
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe — `{"status": "ok"}` |
| `POST` | `/predict` | Classify one ticket |
| `POST` | `/predict/batch` | Classify multiple tickets |
| `GET`  | `/docs` | Swagger UI (auto-generated) |
| `GET`  | `/redoc` | ReDoc UI (auto-generated) |

---

## Request / Response

### `POST /predict`

**Request**
```json
{
  "text": "My internet keeps disconnecting every few minutes."
}
```

**Response `200 OK`**
```json
{
  "category":       "Technical Support",
  "priority":       "High",
  "confidence":     0.9132,
  "routing_reason": "Internet/device issue detected → route to L1 Engineering.",
  "model":          "tfidf-sgd"
}
```

### `POST /predict/batch`

**Request**
```json
{
  "tickets": [
    "I was charged twice this month",
    "Cannot connect to WiFi"
  ]
}
```

**Response `200 OK`**
```json
{
  "results": [
    {
      "category": "Billing",
      "priority": "Medium",
      "confidence": 0.8874,
      "routing_reason": "Billing discrepancy or duplicate charge → route to Accounts.",
      "model": "tfidf-sgd"
    },
    {
      "category": "Technical Support",
      "priority": "High",
      "confidence": 0.9021,
      "routing_reason": "Internet/device issue detected → route to L1 Engineering.",
      "model": "tfidf-sgd"
    }
  ]
}
```

---

## Model Loading Strategy

`backend/predictor.py` uses `functools.lru_cache(maxsize=1)` to load
each joblib model **once per process**. Subsequent requests reuse the
in-memory pipeline — no repeated disk I/O.

```
Request → FastAPI route → predict_single()
                              ↓
                    _load_cat_model()  (lru_cache)
                    _load_pri_model()  (lru_cache, fallback to rule-based)
```

If `model_priority.joblib` is absent the priority prediction
gracefully falls back to a keyword rule-based approach so the service
never returns a 500 for a missing optional artefact.

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and save both models (run once)
python train_model.py

# 3. Start the FastAPI server
uvicorn backend.main:app --reload --port 8000

# 4. Try it
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "my internet keeps disconnecting"}'

# 5. Open interactive docs
start http://localhost:8000/docs
```

---

## Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest httpx

# Run all API tests
pytest test_api.py -v
```

---

## CORS Policy

CORS is configured to `allow_origins=["*"]` for local development.
**Before deploying to production**, restrict this to the specific
Streamlit or frontend origin:

```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.streamlit.app"],
    ...
)
```

---

## Dependency Summary

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework + OpenAPI generation |
| `uvicorn[standard]` | ASGI server |
| `pydantic>=2` | Request/response validation |
| `scikit-learn` | TF-IDF vectoriser + SGD classifier |
| `joblib` | Model serialisation / deserialisation |
| `python-multipart` | Form data support (FastAPI requirement) |

No TensorFlow. No Keras. Works on Streamlit Community Cloud free tier.

---

## Future Improvements

- [ ] Add API-key authentication (`X-API-Key` header)
- [ ] Add `/feedback` endpoint to log corrections for active learning
- [ ] Containerise with Docker (`Dockerfile` + `docker-compose.yml`)
- [ ] Add Prometheus `/metrics` endpoint for monitoring
- [ ] Add request-id middleware for distributed tracing
