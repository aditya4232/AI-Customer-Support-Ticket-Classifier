"""
backend/schemas.py
Pydantic v2 request / response models for the FastAPI classifier API.
"""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


# ── Request models ───────────────────────────────────────────

class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        description="Raw customer support ticket text to classify.",
        examples=["My internet keeps disconnecting every few minutes."],
    )


class BatchRequest(BaseModel):
    tickets: List[str] = Field(
        ...,
        min_length=1,
        description="List of ticket texts to classify in one request.",
        examples=[["I was charged twice", "Cannot connect to WiFi"]],
    )


# ── Response models ──────────────────────────────────────────

class TicketResponse(BaseModel):
    category: str = Field(..., description="Predicted support category.")
    priority: str = Field(..., description="Predicted ticket priority (High/Medium/Low).")
    confidence: float = Field(..., description="Model confidence for the category prediction (0–1).")
    routing_reason: str = Field(..., description="Human-readable routing/escalation instruction.")
    model: str = Field(default="tfidf-sgd", description="Inference engine used.")


class BatchResponse(BaseModel):
    results: List[TicketResponse] = Field(
        ..., description="Ordered list of classification results."
    )


# ── Error model ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
