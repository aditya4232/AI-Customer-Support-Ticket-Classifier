"""
backend/predictor.py
Shared inference logic -- loads scikit-learn joblib pipelines once at
module level and exposes predict_single() / predict_batch().

Priority model uses "ticket_text + predicted_category" as input feature,
mirroring the training strategy in train_model.py.
"""
from __future__ import annotations
import re
import os
import joblib
from functools import lru_cache

# Paths (resolved relative to project root)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAT_MODEL_PATH = os.path.join(_HERE, "model.joblib")
_PRI_MODEL_PATH = os.path.join(_HERE, "model_priority.joblib")


# Text cleaning (mirrors train_model.py)
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Routing reasons (covers all categories in database-tele-data(telecom_cc).csv)
_ROUTING: dict[str, str] = {
    "Technical Support":        "Internet/device issue detected -> route to L1 Engineering.",
    "Billing":                  "Billing discrepancy or duplicate charge -> route to Accounts.",
    "Account Management":       "Account/profile/password issue -> route to Account Services.",
    "Sales / Plan Upgrade":     "Plan upgrade request -> route to Sales team.",
    "General Inquiry":          "General query -> route to Customer Care.",
    "Network Issue":            "Network instability or outage signal -> route to NOC.",
    "Service Disruption":       "Service disruption detected -> escalate to Operations.",
    "Complaints & Escalations": "Customer complaint or repeat issue -> escalate to Senior Support.",
}


def get_routing_reason(category: str) -> str:
    return _ROUTING.get(category, "Standard inquiry -> assigned to the relevant department.")


# Rule-based priority fallback (used when model_priority.joblib is missing)
def _rule_priority(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["not working", "urgent", "down", "broken", "failed", "outage",
                             "complaint", "escalat", "repeat", "disconnecting"]):
        return "High"
    if any(w in t for w in ["refund", "charged", "payment", "bill", "invoice", "account"]):
        return "Medium"
    return "Low"


# Model loading (singleton via lru_cache -- loaded once per process)
@lru_cache(maxsize=1)
def _load_cat_model():
    if not os.path.exists(_CAT_MODEL_PATH):
        raise FileNotFoundError(
            f"Category model not found at {_CAT_MODEL_PATH}. "
            "Run `python train_model.py` first."
        )
    return joblib.load(_CAT_MODEL_PATH)


@lru_cache(maxsize=1)
def _load_pri_model():
    if not os.path.exists(_PRI_MODEL_PATH):
        return None  # graceful degradation -> rule-based fallback
    return joblib.load(_PRI_MODEL_PATH)


# Core prediction
def predict_single(raw_text: str) -> dict:
    """
    Classify a single ticket text.
    Returns a dict matching TicketResponse fields.

    Priority model uses "cleaned_text + category" as feature input,
    mirroring training in train_model.py for best accuracy.
    """
    cleaned = clean_text(raw_text)

    # Step 1: Predict category
    cat_model = _load_cat_model()
    category = cat_model.predict([cleaned])[0]
    confidence = float(cat_model.predict_proba([cleaned]).max())

    # Step 2: Predict priority using text + predicted category as feature
    pri_model = _load_pri_model()
    if pri_model is not None:
        pri_feature = cleaned + " category " + category.lower()
        priority = pri_model.predict([pri_feature])[0]
    else:
        priority = _rule_priority(raw_text)

    return {
        "category":       category,
        "priority":       priority,
        "confidence":     round(confidence, 4),
        "routing_reason": get_routing_reason(category),
        "model":          "tfidf-sgd",
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """Classify a list of ticket texts."""
    return [predict_single(t) for t in texts]
