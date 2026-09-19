from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import psycopg2
import os

app = FastAPI(title="Lend-Tech Risk API", version="1.2.0")

calibrated_model = joblib.load("models/calibrated_pd_model.pkl")
kmeans_model = joblib.load("models/behavioral_cluster.pkl")
cluster_map = joblib.load("models/cluster_map.pkl")
baseline_dist = joblib.load("models/baseline_pd_dist.pkl")

# Database Connection Helper
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "risk_database"),
        user=os.getenv("DB_USER", "risk_admin"),
        password=os.getenv("DB_PASSWORD", "risk_password")
    )

class LoanApplication(BaseModel):
    customer_id: str
    annual_income: float = Field(..., gt=0)
    debt_to_income: float = Field(..., ge=0, le=1)
    loan_amount: float = Field(..., gt=0)
    payment_history_score: float = Field(..., ge=0, le=850)
    missed_payments_6m: int = Field(..., ge=0)
    lgd_assumption: float = Field(default=0.45, ge=0.0, le=1.0)

class RiskDecision(BaseModel):
    probability_of_default: float
    expected_loss: float
    behavioral_segment: str

@app.post("/predict", response_model=RiskDecision)
def evaluate_risk(application: LoanApplication):
    features = pd.DataFrame([{
        'annual_income': application.annual_income,
        'debt_to_income': application.debt_to_income,
        'payment_history_score': application.payment_history_score,
        'missed_payments_6m': application.missed_payments_6m
    }])

    pd_score = float(calibrated_model.predict_proba(features)[0][1])
    expected_loss = float(pd_score * application.lgd_assumption * application.loan_amount)
    
    cluster_idx = int(kmeans_model.predict(features[['payment_history_score', 'missed_payments_6m']])[0])
    segment = cluster_map[cluster_idx]

    # Asynchronous DB write (synchronous for demonstration)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO risk_inferences 
            (customer_id, annual_income, debt_to_income, loan_amount, payment_history_score, missed_payments_6m, probability_of_default, expected_loss, behavioral_segment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (application.customer_id, application.annual_income, application.debt_to_income, application.loan_amount, application.payment_history_score, application.missed_payments_6m, pd_score, expected_loss, segment))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database logging failed: {e}")

    return RiskDecision(probability_of_default=round(pd_score, 4), expected_loss=round(expected_loss, 2), behavioral_segment=segment)
