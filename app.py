import streamlit as st
import pandas as pd
import sqlite3
import json

from pipeline import (
    load_and_prepare_dataset, 
    train_classification_pipeline, 
    custom_tokenizer, 
    log_inference_request, 
    init_database
)
init_database()


st.set_page_config(page_title="Email Classification Pipeline", layout="wide")
st.title("🔒 Automated Email Classification & Audit Logging Pipeline")
st.markdown("---")


st.sidebar.header("⚙️ Pipeline Management")

@st.cache_resource
def get_trained_assets():
    """Trains the model once and caches vectorizer/weights for instant predictions."""
    df = load_and_prepare_dataset()
    vectorizer, model = train_classification_pipeline(df)
    return vectorizer, model

with st.sidebar.spinner("Initializing NLP Engine Assets..."):
    vectorizer, model = get_trained_assets()

st.sidebar.success("Pipeline Online")
st.sidebar.metric(label="Target Pipeline F1-Score", value="97.00%")

def fetch_logs():
    """Extracts operational database table rows ordered by insertion timeline."""
    conn = sqlite3.connect('pipeline_logs.db')
    df = pd.read_sql_query("SELECT * FROM inference_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df


st.subheader("📥 Live Inference Request")

col_sub, col_msg = st.columns(2)
with col_sub:
    email_subject = st.text_input("Email Subject", placeholder="e.g., URGENT: Action Required")
with col_msg:
    email_message = st.text_area("Email Body / Message Content", placeholder="Type or paste the email contents here...")

if st.button("Run Real-Time Classification", type="primary"):
    if not email_subject.strip() and not email_message.strip():
        st.warning("Please input text data metrics before triggering inference classification.")
    else:
        combined_text = f"{email_subject} {email_message}"
        tokens = custom_tokenizer(combined_text)
        
        vectorized_text = vectorizer.transform([combined_text])
        prediction_id = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        
        confidence = probabilities[prediction_id]
        prediction_label = "Malicious" if prediction_id == 1 else "Safe"
        
        log_inference_request(
            subject=email_subject,
            message=email_message,
            tokens=tokens,
            prediction=prediction_label,
            confidence=confidence
        )
        
        st.markdown("### 📊 Inference Engine Output")
        c1, c2, c3 = st.columns(3)
        
        if prediction_label == "Malicious":
            c1.error(f"Prediction: **{prediction_label}**")
        else:
            c1.success(f"Prediction: **{prediction_label}**")
            
        c2.metric("Confidence Score", f"{confidence:.2%}")
        c3.metric("Tokens Extracted", len(tokens))
        
        st.info(f"**Tokenized Metadata (JSON Format for Auditing):** \n `{json.dumps(tokens)}`")


st.markdown("---")
st.subheader("📋 Secure System Audit Logs (Real-Time DB Mirror)")

logs_df = fetch_logs()

if not logs_df.empty:
    st.dataframe(
        logs_df, 
        column_config={
            "log_id": "ID",
            "timestamp": "Timestamp",
            "raw_subject": "Subject",
            "raw_message": "Message Body",
            "tokenized_metadata": "Tokenized Metadata",
            "prediction_label": "Classification Result",
            "confidence_score": st.column_config.NumberColumn("Confidence", format="%.4f")
        }, 
        use_container_width=True,
        hide_index=True
    )
    
    if st.button("Clear Audit Trail Database"):
        conn = sqlite3.connect('pipeline_logs.db')
        conn.execute("DELETE FROM inference_logs")
        conn.commit()
        conn.close()
        st.rerun()
else:
    st.info("No inference logs found in the database. Run a classification above to generate data logs.")
