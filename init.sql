CREATE TABLE IF NOT EXISTS risk_inferences (
    inference_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    annual_income NUMERIC,
    debt_to_income NUMERIC,
    loan_amount NUMERIC,
    payment_history_score NUMERIC,
    missed_payments_6m INTEGER,
    probability_of_default NUMERIC,
    expected_loss NUMERIC,
    behavioral_segment VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
