"""
backend/main.py
FastAPI application — AI Customer Support Ticket Classifier API.

Run:
    uvicorn backend.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette import status

from backend.schemas import (
    TicketRequest,
    TicketResponse,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
)
from backend.predictor import predict_single, predict_batch

# ── App init ─────────────────────────────────────────────────
app = FastAPI(
    title="AI Support Ticket Classifier",
    description=(
        "REST API for classifying telecom customer-support tickets.\n\n"
        "Powered by **TF-IDF + SGD** (scikit-learn). "
        "No TensorFlow dependency — lightweight and cloud-friendly."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow all origins in dev; tighten in production) ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ─────────────────────────────────────────────
@app.get(
    "/health",
    summary="Liveness check",
    response_description="Service status",
    tags=["Utility"],
)
def health() -> dict:
    """Returns `{"status": "ok"}` when the service is running."""
    return {"status": "ok"}


# ── Single prediction ────────────────────────────────────────
@app.post(
    "/predict",
    response_model=TicketResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Classify a single support ticket",
    tags=["Classification"],
)
def predict(request: TicketRequest) -> TicketResponse:
    """
    Accept a customer support ticket description and return:
    - **category** — e.g. *Technical Support*, *Billing*, *Network Issue*
    - **priority** — *High*, *Medium*, or *Low*
    - **confidence** — model probability for the predicted category
    - **routing_reason** — plain-English escalation instruction
    - **model** — inference engine used (`tfidf-sgd`)
    """
    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Ticket text must not be blank.",
        )
    try:
        result = predict_single(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc

    return TicketResponse(**result)


# ── Batch prediction ─────────────────────────────────────────
@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Classify multiple tickets in one request",
    tags=["Classification"],
)
def predict_batch_endpoint(request: BatchRequest) -> BatchResponse:
    """
    Accept a list of ticket texts and return a corresponding list of
    classification results. Results are ordered to match input order.
    """
    # Strip and validate each ticket
    cleaned_tickets = [t.strip() for t in request.tickets]
    empty = [i for i, t in enumerate(cleaned_tickets) if not t]
    if empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Blank ticket text at index(es): {empty}",
        )

    try:
        results = predict_batch(cleaned_tickets)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc

    return BatchResponse(results=[TicketResponse(**r) for r in results])
