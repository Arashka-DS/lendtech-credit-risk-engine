from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Lend-Tech Risk API", version="1.1.0")

# Load artifacts
calibrated_model = joblib.load("models/calibrated_pd_model.pkl")
kmeans_model = joblib.load("models/behavioral_cluster.pkl")
cluster_map = joblib.load("models/cluster_map.pkl")
baseline_dist = joblib.load("models/baseline_pd_dist.pkl")

class LoanApplication(BaseModel):
    customer_id: str
    annual_income: float = Field(..., gt=0)
    debt_to_income: float = Field(..., ge=0, le=1)
    loan_amount: float = Field(..., gt=0)
    payment_history_score: float = Field(..., ge=0, le=850)
    missed_payments_6m: int = Field(..., ge=0)
    lgd_assumption: float = Field(default=0.45, ge=0.0, le=1.0) # Loss Given Default (Standard 45%)

class RiskDecision(BaseModel):
    customer_id: str
    probability_of_default: float
    expected_loss: float
    behavioral_segment: str

@app.post("/predict", response_model=RiskDecision)
def evaluate_risk(application: LoanApplication):
    # Construct feature array
    features = pd.DataFrame([{
        'annual_income': application.annual_income,
        'debt_to_income': application.debt_to_income,
        'payment_history_score': application.payment_history_score,
        'missed_payments_6m': application.missed_payments_6m
    }])

    # 1. Calibrated Probability of Default (PD)
    pd_score = calibrated_model.predict_proba(features)[0][1]

    # 2. Expected Loss Calculation (PD * LGD * EAD)
    ead = application.loan_amount
    lgd = application.lgd_assumption
    expected_loss = pd_score * lgd * ead

    # 3. Behavioral Clustering
    behavior = features[['payment_history_score', 'missed_payments_6m']]
    cluster_idx = kmeans_model.predict(behavior)[0]
    segment = cluster_map[cluster_idx]

    return RiskDecision(
        customer_id=application.customer_id,
        probability_of_default=round(pd_score, 4),
        expected_loss=round(expected_loss, 2),
        behavioral_segment=segment
    )

@app.post("/metrics/psi")
def calculate_psi(recent_pd_scores: list[float]):
    """Calculate Population Stability Index to detect data drift."""
    if not recent_pd_scores:
        raise HTTPException(status_code=400, detail="Must provide recent scores.")
    
    recent_bins = np.histogram(recent_pd_scores, bins=10, range=(0, 1))[0]
    recent_dist = recent_bins / len(recent_pd_scores)
    
    # Avoid division by zero
    epsilon = 1e-4
    recent_dist = np.where(recent_dist == 0, epsilon, recent_dist)
    baseline_safe = np.where(baseline_dist == 0, epsilon, baseline_dist)
    
    psi_values = (recent_dist - baseline_safe) * np.log(recent_dist / baseline_safe)
    total_psi = np.sum(psi_values)
    
    drift_status = "Stable" if total_psi < 0.1 else ("Warning" if total_psi < 0.25 else "Critical Drift")
    
    return {"PSI": round(total_psi, 4), "status": drift_status}
