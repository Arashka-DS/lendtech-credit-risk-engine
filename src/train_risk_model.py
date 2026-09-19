import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.stats import ks_2samp
import joblib
import os

np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    'annual_income': np.random.normal(50000, 15000, n_samples),
    'debt_to_income': np.random.uniform(0.1, 0.8, n_samples),
    'payment_history_score': np.random.uniform(300, 850, n_samples),
    'missed_payments_6m': np.random.poisson(1, n_samples)
})

risk_score = (data['debt_to_income'] * 2) - (data['payment_history_score'] / 1000) + (data['missed_payments_6m'] * 0.5)
data['default'] = (risk_score > np.median(risk_score) + 0.2).astype(int)

X = data.drop('default', axis=1)
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost Risk Model...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss')
xgb.fit(X_train, y_train)
raw_probs = xgb.predict_proba(X_test)[:, 1]

print("Applying Isotonic Probability Calibration...")
calibrated_xgb = CalibratedClassifierCV(estimator=xgb, method='isotonic', cv='prefit')
calibrated_xgb.fit(X_test, y_test)
calibrated_probs = calibrated_xgb.predict_proba(X_test)[:, 1]

# Quantitative Rigor Metrics
brier = brier_score_loss(y_test, calibrated_probs)
roc_auc = roc_auc_score(y_test, calibrated_probs)

# K-S Statistic Calculation
def calculate_ks(y_true, y_prob):
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    defaults = df[df['true'] == 1]['prob']
    non_defaults = df[df['true'] == 0]['prob']
    ks_stat, _ = ks_2samp(defaults, non_defaults)
    return ks_stat

ks_statistic = calculate_ks(y_test, calibrated_probs)

print(f"Calibration Complete. Brier Score: {brier:.4f} | ROC-AUC: {roc_auc:.4f} | K-S Stat: {ks_statistic:.4f}")

print("Fitting Behavioral K-Means Cluster...")
behavior_features = data[['payment_history_score', 'missed_payments_6m']]
kmeans = KMeans(n_clusters=3, random_state=42).fit(behavior_features)
cluster_map = {0: "Consistent Borrowers", 1: "Overleveraged / Late", 2: "High-Risk Defaulters"}

baseline_bins = np.histogram(calibrated_probs, bins=10, range=(0, 1))[0]
baseline_dist = baseline_bins / len(calibrated_probs)

os.makedirs("models", exist_ok=True)
joblib.dump(calibrated_xgb, "models/calibrated_pd_model.pkl")
joblib.dump(kmeans, "models/behavioral_cluster.pkl")
joblib.dump(baseline_dist, "models/baseline_pd_dist.pkl")
joblib.dump(cluster_map, "models/cluster_map.pkl")
