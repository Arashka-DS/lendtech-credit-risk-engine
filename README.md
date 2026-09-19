# lendtech-credit-risk-engine
End-to-end FinTech credit risk platform. Features an isotonic-calibrated XGBoost model for Probability of Default (PD), Expected Loss (EL) calculations, Population Stability Index (PSI) monitoring, and a containerized FastAPI backend with background logging / inferences to PostgreSQL for Metabase/Power BI visualization.

# Lend-Tech Credit Default Scoring & Risk Engine

An end-to-end FinTech credit risk platform combining statistical rigor with modern data engineering. This project calculates true calibrated Probability of Default (PD) and Expected Loss (EL), tracks behavioral clustering, and persists inference data for executive BI visualization.

## Architecture & Stack
- **Modeling:** XGBoost + Scikit-learn (Isotonic Calibration, K-Means Clustering, K-S Statistic).
- **Backend:** FastAPI with strict Pydantic financial guardrails.
- **Data Engineering:** PostgreSQL for automated inference logging in the background.
- **Visualization:** Metabase & Power BI (via JSON dark-mode templates).
- **Infrastructure:** Fully containerized via Docker Compose.

## Key Financial Metrics Implemented
- **Expected Loss (EL):** $EL = PD \times LGD \times EAD$
- **Probability Calibration:** Isotonic regression guarantees predicted scores map directly to real-world default frequencies.
- **Data Drift (PSI):** Population Stability Index calculation detects shifts in inference data distributions against the training baseline.

## Quick Start
1. Clone the repository.
2. Generate synthetic data and train the model: `python src/train_risk_model.py`
3. Spin up the infrastructure: `docker-compose up -d --build`
4. Access the API at `http://localhost:8000/docs` to submit loan applications.
5. Access Metabase at `http://localhost:3000` to visualize the PostgreSQL risk ledger.
