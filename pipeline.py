import re
import json
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def custom_tokenizer(text):
    if not text:
        return []
    text=text.lower()
    text=re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' [URL] ', text)
    text=re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' [EMAIL] ', text)
    tokens=re.findall(r'\b\w+\b|[!$]', text)
    return [t for t in tokens if len(t) > 1 or t in ['!', '$']]


def load_and_prepare_dataset():
    data=[
        {"subject": "Q3 Financial Review", "message": "Hi Team, please find attached the spreadsheet for the upcoming Q3 review.", "label": "Safe"},
        {"subject": "Project Pipeline Update", "message": "The deployment is complete. Streamlit frontend is up and running smoothly.", "label": "Safe"},
        {"subject": "Weekly Sync Meeting", "message": "Let's review the MySQL database indexing architecture this afternoon at 2 PM.", "label": "Safe"},
        {"subject": "Documentation guidelines", "message": "Please log all tokenized metadata fields inside our internal registry updates.", "label": "Safe"},
        {"subject": "URGENT: Password Reset Required", "message": "Action required! Secure your account immediately by clicking http://enron-verify.com now!", "label": "Malicious"},
        {"subject": "Invoice Overdue - Immediate Action", "message": "Your payment of $4,500 is overdue. Please log in here to avoid legal penalties.", "label": "Malicious"},
        {"subject": "Exclusive Account Verification Offer", "message": "Get priority access! Verify your login credentials to win $1000 cash bonus!", "label": "Malicious"},
        {"subject": "Security Breach Notice", "message": "Suspicious login detected on your server profile. Click to update configuration.", "label": "Malicious"}
    ]

    df=pd.DataFrame(data * 25) 
    df['full_text']=df['subject'] + " " + df['message']
    return df

def train_classification_pipeline(df):
    vectorizer=TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None, max_features=500)
    X=vectorizer.fit_transform(df['full_text'])
    y=df['label'].apply(lambda x: 1 if x == 'Malicious' else 0)
    
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    
    model=LogisticRegression(C=5.0, class_weight='balanced')
    model.fit(X_train, y_train)
    
    preds=model.predict(X_test)
    score=f1_score(y_test, preds)
    print(f"🌟 Model Pipeline Initialization Complete. Achieved F1-Score: {score:.2%}")
    
    return vectorizer, model


def init_database():

    conn=sqlite3.connect('pipeline_logs.db')
    cursor=conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_subject TEXT,
            raw_message TEXT,
            tokenized_metadata TEXT,
            prediction_label TEXT,
            confidence_score REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_inference_request(subject, message, tokens, prediction, confidence):
    conn=sqlite3.connect('pipeline_logs.db')
    cursor=conn.cursor()
    cursor.execute('''
        INSERT INTO inference_logs (raw_subject, raw_message, tokenized_metadata, prediction_label, confidence_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (subject, message, json.dumps(tokens), prediction, float(confidence)))
    conn.commit()
    conn.close()
