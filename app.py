import streamlit as st
import pandas as pd
import pickle
import re

# Page config
st.set_page_config(page_title="Ticket Support AI", layout="wide")

# Load models efficiently
@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        vectorizer, model_category, model_priority = pickle.load(f)
    return vectorizer, model_category, model_priority

vectorizer, model_category, model_priority = load_models()

# Load training data for analytics
@st.cache_data
def load_data():
    df = pd.read_csv("telecom_cc.csv")
    df.columns = ["text", "category", "priority"]
    return df

df = load_data()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    category = model_category.predict(X)[0]
    priority = model_priority.predict(X)[0]
    
    # Calculate confidence / probability
    cat_probs = model_category.predict_proba(X)[0]
    prob = max(cat_probs)

    return category, priority, prob

def get_rule_reason(category):
    # Determine the strict reason offline without external AI
    reasons = {
        "Technical Support": "Internet connectivity or hardware issue affecting service usage.",
        "Billing": "Duplicate payment detected or invoice discrepancy in subscription billing.",
        "Account Management": "User unable to access account due to password or profile issue.",
        "Sales / Plan Upgrade": "Customer requesting product upgrade or plan evaluation.",
        "Sales": "Customer requesting product upgrade or sales information."
    }
    return reasons.get(category, "Standard inquiry assigned to the designated department.")

# ----------------- UI -----------------
st.title("🎫 Support Ticket Routing Engine")
st.markdown("Automated ticket classification using Naive Bayes strictly trained on `telecom_cc.csv`.")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Submit a Ticket")
    user_input = st.text_area("Describe the customer issue:", height=150, placeholder="e.g., I was charged twice for my monthly subscription this month...")

    if st.button("Classify Ticket", type="primary"):
        if user_input.strip() == "":
            st.error("Please enter a valid ticket description.")
        else:
            category, priority, prob = predict(user_input)
            reason = get_rule_reason(category)

            st.success("Ticket Successfully Analyzed!")
            
            # Show interactive metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Category", category)
            m2.metric("Priority", priority)
            m3.metric("Confidence Score", f"{prob:.1%}")
            
            st.info(f"**System Routing Reason:** {reason}")

with col2:
    st.subheader("Training Data Insights")
    st.markdown(f"**Total Historical Tickets:** {len(df)}")
    
    st.markdown("**Volume by Category:**")
    st.bar_chart(df["category"].value_counts(), height=200)
    
    st.markdown("**Volume by Priority:**")
    st.bar_chart(df["priority"].value_counts(), color="#ffaa00", height=200)
