-- Card 1: Total Portfolio Value at Risk (Number Card)

SELECT 
    SUM(expected_loss) AS total_value_at_risk 
FROM loan_predictions
WHERE timestamp >= NOW() - INTERVAL '30 days';

-- Card 2: Daily Credit Decision Matrix (Stacked Bar Chart)

SELECT 
    DATE(timestamp) AS decision_date, 
    decision, 
    COUNT(*) as volume
FROM loan_predictions
GROUP BY 1, 2 
ORDER BY 1 DESC;

-- Card 3: Real-Time Probability of Default (PD) Drift (Line Chart with Trendline)

SELECT 
    DATE_TRUNC('hour', timestamp) AS time_window,
    AVG(probability_of_default) * 100 AS avg_portfolio_pd
FROM loan_predictions
GROUP BY 1
ORDER BY 1 ASC;

-- Card 4: High-Risk Application Audit (Data Table)

SELECT 
    application_id,
    requested_amount,
    probability_of_default,
    expected_loss,
    decision,
    timestamp
FROM loan_predictions
WHERE probability_of_default > 0.10
ORDER BY probability_of_default DESC
LIMIT 50;
