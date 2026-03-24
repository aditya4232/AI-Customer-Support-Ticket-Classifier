"""
app.py  –  AI Customer Support Ticket Classifier
=================================================
Streamlit UI with three selectable inference engines:
  • TF-IDF + Naive Bayes
  • Neural Network (Embedding + Dense)
  • LSTM (Bidirectional)
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="🎫 AI Ticket Classifier — Telecom",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  .hero-title {
    font-size: 2.6rem; font-weight: 800; text-align: center;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    text-align: center; color: #9ca3af; font-size: 0.95rem; margin-bottom: 1.4rem;
  }
  .result-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #1c1533 100%);
    border: 1px solid #4f46e5; border-radius: 16px;
    padding: 1.4rem 1.8rem; margin-top: 1rem;
  }
  .badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 9999px; font-size: 0.8rem; font-weight: 700;
    margin-right: 0.4rem;
  }
  .badge-cat  { background:#312e81; color:#c7d2fe; border:1px solid #6366f1; }
  .badge-pri  { background:#1e3a5f; color:#bfdbfe; border:1px solid #3b82f6; }
  .badge-conf { background:#14532d; color:#bbf7d0; border:1px solid #22c55e; }
  .section-card {
    background: #1a1d27; border: 1px solid #2d2f3e;
    border-radius:12px; padding:1.2rem 1.5rem; margin-top:0.6rem;
  }
</style>
""", unsafe_allow_html=True)

DATA_FILE    = "database-tele-data(telecom_cc).csv"
MAX_LEN      = 100


# ─── Helpers ────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Model loaders (cached) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nb_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)           # (tfidf, nb_cat, nb_pri)

@st.cache_resource(show_spinner=False)
def load_nb_label_encoders():
    with open("label_encoders.pkl", "rb") as f:
        return pickle.load(f)           # (le_cat, le_pri)

@st.cache_resource(show_spinner=False)
def load_keras_assets():
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        le_cat, le_pri = pickle.load(f)
    return tok, le_cat, le_pri

@st.cache_resource(show_spinner=False)
def load_nn_model():
    import tensorflow as tf
    cat_m = tf.keras.models.load_model("nn_model/category_model.h5")
    pri_m = tf.keras.models.load_model("nn_model/priority_model.h5")
    return cat_m, pri_m

@st.cache_resource(show_spinner=False)
def load_lstm_model():
    import tensorflow as tf
    cat_m = tf.keras.models.load_model("lstm_model/category_model.h5")
    pri_m = tf.keras.models.load_model("lstm_model/priority_model.h5")
    return cat_m, pri_m

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.rename(columns={df.columns[0]: "text",
                              df.columns[1]: "category",
                              df.columns[2]: "priority"})
    df = df.dropna(subset=["text", "category", "priority"])
    return df

# ─── Inference functions ────────────────────────────────────
def predict_nb(text: str):
    tfidf, nb_cat, nb_pri = load_nb_model()
    le_cat, le_pri = load_nb_label_encoders()
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    cat_enc = nb_cat.predict(X)[0]
    pri_enc = nb_pri.predict(X)[0]
    prob    = max(nb_cat.predict_proba(X)[0])
    cat     = le_cat.inverse_transform([cat_enc])[0]
    pri     = le_pri.inverse_transform([pri_enc])[0]
    return cat, pri, float(prob)


def predict_nn(text: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tok, le_cat, le_pri = load_keras_assets()
    nn_cat, nn_pri = load_nn_model()
    cleaned = clean_text(text)
    seq = pad_sequences(tok.texts_to_sequences([cleaned]),
                        maxlen=MAX_LEN, padding="post", truncating="post")
    cat_probs = nn_cat.predict(seq, verbose=0)[0]
    pri_probs = nn_pri.predict(seq, verbose=0)[0]
    cat   = le_cat.inverse_transform([np.argmax(cat_probs)])[0]
    pri   = le_pri.inverse_transform([np.argmax(pri_probs)])[0]
    prob  = float(np.max(cat_probs))
    return cat, pri, prob


def predict_lstm(text: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tok, le_cat, le_pri = load_keras_assets()
    lstm_cat, lstm_pri = load_lstm_model()
    cleaned = clean_text(text)
    seq = pad_sequences(tok.texts_to_sequences([cleaned]),
                        maxlen=MAX_LEN, padding="post", truncating="post")
    cat_probs  = lstm_cat.predict(seq, verbose=0)[0]
    pri_probs  = lstm_pri.predict(seq, verbose=0)[0]
    cat  = le_cat.inverse_transform([np.argmax(cat_probs)])[0]
    pri  = le_pri.inverse_transform([np.argmax(pri_probs)])[0]
    prob = float(np.max(cat_probs))
    return cat, pri, prob


def get_routing_reason(category: str) -> str:
    reasons = {
        "Technical Support":   "Internet/device issue detected → route to L1 Engineering.",
        "Billing":             "Billing discrepancy or duplicate charge → route to Accounts.",
        "Account Management":  "Account/profile/password issue → route to Account Services.",
        "Sales / Plan Upgrade":"Plan upgrade request → route to Sales team.",
        "General Inquiry":     "General query → route to Customer Care.",
        "Network Issue":       "Network instability or outage signal → route to NOC.",
        "Service Disruption":  "Service disruption detected → escalate to Operations.",
    }
    return reasons.get(category, "Standard inquiry → assigned to the relevant department.")


def get_priority_color(priority: str) -> str:
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(priority, "⚪")


# ─── Load data ──────────────────────────────────────────────
df = load_data()


# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    model_choice = st.selectbox(
        "Select Inference Engine",
        ["TF-IDF + Naive Bayes"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.metric("Total Tickets", len(df))
    st.metric("Categories", df["category"].nunique())
    st.metric("Priorities", df["priority"].nunique())
    st.markdown("---")
    st.markdown("**Category Distribution**")
    st.bar_chart(df["category"].value_counts(), height=180)
    st.markdown("**Priority Distribution**")
    st.bar_chart(df["priority"].value_counts(), color="#a855f7", height=140)


# ─── Hero header ────────────────────────────────────────────
st.markdown("<div class='hero-title'>🎫 AI Support Ticket Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Telecom ticket routing engine powered by TF-IDF + Naive Bayes</div>", unsafe_allow_html=True)

# ─── Model info badge ───────────────────────────────────────
model_meta = {
    "TF-IDF + Naive Bayes":             ("🔵 Naive Bayes",  "#1d4ed8", "Fast & interpretable baseline using TF-IDF bigrams."),
}
label, color, desc = model_meta[model_choice]
st.markdown(f"""
<div style='background:{color}22;border:1px solid {color}55;
            border-radius:10px;padding:0.6rem 1rem;margin-bottom:1rem;'>
  <strong style='color:{color}'>{label}</strong>
  &nbsp;·&nbsp;<span style='color:#d1d5db'>{desc}</span>
</div>
""", unsafe_allow_html=True)

# ─── Main layout ────────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("📝 Submit a Support Ticket")
    ticket_text = st.text_area(
        "Describe the customer issue:",
        height=160,
        placeholder="e.g. I was charged twice for my monthly plan this billing cycle...",
    )

    if st.button("🔍  Classify Ticket", type="primary", width="stretch"):
        if not ticket_text.strip():
            st.error("Please enter a valid ticket description.")
        else:
            with st.spinner(f"Running {model_choice} inference…"):
                try:
                    if model_choice == "TF-IDF + Naive Bayes":
                        if not os.path.exists("model.pkl"):
                            st.error("model.pkl not found. Run `python train_advanced.py` first.")
                            st.stop()
                        category, priority, prob = predict_nb(ticket_text)

                    reason = get_routing_reason(category)
                    p_icon = get_priority_color(priority)

                    st.success("✅  Ticket classified successfully!")

                    # ─ Metrics row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🏷️ Category", category)
                    m2.metric(f"{p_icon} Priority", priority)
                    m3.metric("📊 Confidence", f"{prob:.1%}")

                    # ─ Result card
                    st.markdown(f"""
<div class='result-card'>
  <p style='color:#e2e8f0;margin-bottom:0.5rem;'>
    <strong>📌 Routing Decision</strong>
  </p>
  <p style='color:#c4b5fd;font-size:1rem;'>{reason}</p>
  <hr style='border-color:#3730a3;margin:0.8rem 0'>
  <span class='badge badge-cat'>Category: {category}</span>
  <span class='badge badge-pri'>Priority: {priority}</span>
  <span class='badge badge-conf'>Confidence: {prob:.1%}</span>
</div>
""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.info("Tip: Make sure you have run `python train_advanced.py` to build all models.")


with col2:
    st.subheader("📈 Live Analytics")

    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("**Top Categories**")
        st.bar_chart(df["category"].value_counts(), height=200)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("**Priority Breakdown**")
        st.bar_chart(df["priority"].value_counts(), color="#f43f5e", height=160)
        st.markdown("</div>", unsafe_allow_html=True)

    # ─ Category × Priority heatmap
    pivot = df.pivot_table(index="category", columns="priority",
                            aggfunc="size", fill_value=0)
    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("**Category × Priority Matrix**")
        st.dataframe(pivot, width=700)
        st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with TensorFlow · Keras · scikit-learn · Streamlit  |  Dataset: database-tele-data(telecom_cc).csv")
