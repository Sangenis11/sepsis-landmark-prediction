"""
Evaluation: Calibration Analysis for RF Balanced Ensemble (Landmark 18h)

This script evaluates probabilistic calibration of the RF balanced
ensemble model using:
• Brier score
• Calibration curve
• Calibration slope and intercept
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ============================================================
# USER CONFIG
# ============================================================

DATA_PATH = "data/processed/V02_cohort_ANALYSIS_READY_noBMI.csv"
MODEL_DIR = "outputs/model_results/balanced_rf_ensemble"
LANDMARK = 18
K = 40  # Must match training

OUT_DIR = "outputs/evaluation/rf_lm18_calibration"
os.makedirs(OUT_DIR, exist_ok=True)

OUTCOME = "sepsis_next_6h"
PATIENT_ID = "subject_id"

NUMERIC_FEATURES = [
    "heart_rate_max", "spo2_min", "temperature_max",
    "rr_max", "map_ni", "anchor_age"
]

BINARY_FEATURES = [
    "vasopressor_mean", "crrt_mean", "invasive_mean",
    "noninvasive_mean", "highflow_mean",
    "gcs_min_missing", "map_ni_missing"
]

CATEGORICAL_FEATURES = [
    "gender", "race_grouped",
    "gcs_min_cat_num", "elixhauser_cat_num"
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# ============================================================
# LOAD TEST SET
# ============================================================

df = pd.read_csv(DATA_PATH)
patients = df[PATIENT_ID].unique()
_, test_pats = train_test_split(patients, test_size=0.25, random_state=42)

df_test = df[df[PATIENT_ID].isin(test_pats)]
df_test_lm = df_test[df_test["landmark_hr"] == LANDMARK]

X_test = df_test_lm[ALL_FEATURES]
y_true = df_test_lm[OUTCOME]

# ============================================================
# ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(lm, model_dir, K, X_test):
    lm_dir = os.path.join(model_dir, f"LM{lm}")
    probs_list = []

    for i in range(1, K + 1):
        files = [f for f in os.listdir(lm_dir) if f.endswith(f"{i:02d}.joblib")]
        pipe = joblib.load(os.path.join(lm_dir, files[0]))
        probs_list.append(pipe.predict_proba(X_test)[:, 1])

    return np.mean(np.vstack(probs_list), axis=0)

y_prob = ensemble_predict(LANDMARK, MODEL_DIR, K, X_test)

# ============================================================
# CALIBRATION METRICS
# ============================================================

brier = brier_score_loss(y_true, y_prob)

lr_cal = LogisticRegression(solver="lbfgs")
lr_cal.fit(y_prob.reshape(-1, 1), y_true)

calibration_results = {
    "Brier_score": brier,
    "Calibration_intercept": lr_cal.intercept_[0],
    "Calibration_slope": lr_cal.coef_[0][0]
}

pd.DataFrame([calibration_results]).to_csv(
    os.path.join(OUT_DIR, "Calibration_RF_LM18.csv"),
    index=False
)

print("Calibration metrics:", calibration_results)

# ============================================================
# CALIBRATION CURVE
# ============================================================

prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="RF Ensemble")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed event rate")
plt.title("Calibration Curve — RF Ensemble (LM18)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Calibration_Curve_RF_LM18.png"), dpi=300)
plt.close()

print("\n✅ Calibration evaluation complete.")

