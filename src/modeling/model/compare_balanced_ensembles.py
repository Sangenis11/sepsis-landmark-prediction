"""
Model Comparison: LR vs RF vs XGB Balanced Ensembles

This script loads saved sub-models from the three balanced ensemble approaches
(Logistic Regression, Random Forest, XGBoost), generates test predictions,
and compares performance across landmarks.

Outputs:
• Full performance table
• Best model per landmark (by AUROC)
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ============================================================
# USER CONFIG — WHERE YOUR DATA LIVES
# ============================================================

DATA_PATH = "data/processed/V02_cohort_ANALYSIS_READY_noBMI.csv"

# Model output folders from previous scripts
OUT_DIR_LR  = "outputs/model_results/balanced_lr_ensemble"
OUT_DIR_RF  = "outputs/model_results/balanced_rf_ensemble"
OUT_DIR_XGB = "outputs/model_results/balanced_xgb_ensemble"

SAVE_DIR = "outputs/model_comparison"
os.makedirs(SAVE_DIR, exist_ok=True)

LANDMARKS = [6, 12, 18, 24]
OUTCOME = "sepsis_next_6h"
PATIENT_ID = "subject_id"

K_DICT = {6: 20, 12: 30, 18: 40, 24: 50}  # Must match training

# ============================================================
# FEATURE SET (must match training scripts)
# ============================================================

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
# METRIC FUNCTION
# ============================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) else 0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

# ============================================================
# LOAD DATA + RECREATE TEST SPLIT
# ============================================================

df = pd.read_csv(DATA_PATH)

patients = df[PATIENT_ID].unique()
train_pats, test_pats = train_test_split(patients, test_size=0.25, random_state=42)

df_test = df[df[PATIENT_ID].isin(test_pats)]

# ============================================================
# ENSEMBLE PREDICTION LOADER
# ============================================================

def ensemble_predict(lm, model_dir, K):

    lm_dir = os.path.join(model_dir, f"LM{lm}")
    if not os.path.exists(lm_dir):
        raise FileNotFoundError(f"Model folder not found: {lm_dir}")

    probs_list = []

    for i in range(1, K + 1):

        # Match filenames like xgb_model_01.joblib, rf_model_01.joblib, etc.
        files = [f for f in os.listdir(lm_dir) if f.endswith(f"{i:02d}.joblib")]
        if not files:
            raise FileNotFoundError(f"Missing submodel {i:02d} in {lm_dir}")

        pipe = joblib.load(os.path.join(lm_dir, files[0]))

        X_test = df_test[df_test["landmark_hr"] == lm][ALL_FEATURES]
        prob = pipe.predict_proba(X_test)[:, 1]
        probs_list.append(prob)

    return np.mean(np.vstack(probs_list), axis=0)

# ============================================================
# RUN COMPARISON
# ============================================================

results = []

for lm in LANDMARKS:
    print(f"\n🔍 Evaluating ensembles at Landmark {lm}h")

    y_true = df_test[df_test["landmark_hr"] == lm][OUTCOME]

    # Logistic Regression Ensemble
    y_prob_lr = ensemble_predict(lm, OUT_DIR_LR, K_DICT[lm])
    m_lr = compute_metrics(y_true, y_prob_lr)
    m_lr.update({"model": "LR", "landmark_hr": lm})
    results.append(m_lr)

    # Random Forest Ensemble
    y_prob_rf = ensemble_predict(lm, OUT_DIR_RF, K_DICT[lm])
    m_rf = compute_metrics(y_true, y_prob_rf)
    m_rf.update({"model": "RF", "landmark_hr": lm})
    results.append(m_rf)

    # XGBoost Ensemble
    y_prob_xgb = ensemble_predict(lm, OUT_DIR_XGB, K_DICT[lm])
    m_xgb = compute_metrics(y_true, y_prob_xgb)
    m_xgb.update({"model": "XGB", "landmark_hr": lm})
    results.append(m_xgb)

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)[[
    "landmark_hr", "model", "AUROC", "AUPRC", "Accuracy",
    "Sensitivity", "Specificity", "Precision", "Recall", "F1"
]]

best_models = results_df.loc[results_df.groupby("landmark_hr")["AUROC"].idxmax()]

results_df.to_csv(os.path.join(SAVE_DIR, "Comparison_LR_RF_XGB_Performance.csv"), index=False)
best_models.to_csv(os.path.join(SAVE_DIR, "BestModel_PerLandmark.csv"), index=False)

print("\n✅ Model comparison complete")
print(results_df)
print("\n🏆 Best model per landmark (AUROC):")
print(best_models)

