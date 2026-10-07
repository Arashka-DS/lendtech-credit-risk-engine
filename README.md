# Lend-Tech Credit Default Scoring & Risk Engine

An end-to-end FinTech credit risk platform combining statistical rigor with modern data engineering. This project calculates true calibrated Probability of Default (PD) and Expected Loss (EL), tracks behavioral clustering, and persists inference data for real-time operational and executive BI visualization.

## 🏛️ Architecture & Stack
* **Modeling:** XGBoost + Scikit-learn (Isotonic Calibration, K-Means Clustering, K-S Statistic).
* **Backend:** FastAPI with strict Pydantic financial guardrails.
* **Data Engineering:** PostgreSQL for automated background inference logging.
* **Visualization (Dual-Layer BI):** Metabase (Operational) & Power BI (Executive).
* **Infrastructure:** Fully containerized via Docker Compose.

## 📊 Key Financial Metrics Implemented
* **Expected Loss (EL):** `EL = PD × LGD × EAD`
* **Probability Calibration:** Isotonic regression guarantees predicted scores map directly to real-world default frequencies.
* **Data Drift (PSI):** Population Stability Index calculation detects shifts in inference data distributions against the training baseline.

## 🚀 Quick Start
1. **Clone the repository.**
2. (Optional) **Generate synthetic data and train the model** (As it will run as the first task inside the container too): 
   ```bash
   python src/train_risk_model.py
   ```
3. **Spin up the infrastructure:**
   ```bash
   docker-compose up -d --build
   ```
4. **Test the Pipeline**: Access the interactive API at `http://localhost:8000/docs` to submit synthetic loan applications.
5. **Operational BI**: Access Metabase at `http://localhost:3000` to visualize the PostgreSQL risk ledger in real-time.
6. **Executive BI**: Import `bi_vis_misc/PowerBI_FinTech_Theme.json` into Power BI Desktop and apply the DAX scripts found in the `bi_vis_misc` folder to the connected PostgreSQL database. There are also sql queries for metabase cards in the `metabase.sql`.
