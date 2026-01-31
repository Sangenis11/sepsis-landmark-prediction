"""
Interpretability: Permutation Feature Importance
RF Balanced Ensemble — Landmark 18h

This script computes permutation importance averaged
across all RF ensemble submodels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ============================================================
# USER CONFIG
# ============================================================

DATA_PATH = "..._READY_noBMI.csv"
MODEL_DIR = "outputs/model_results/balanced_rf_ensemble"
LANDMARK = 18
K = 40

OUT_DIR = "outputs/interpretability/rf_lm18"
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
# PERMUTATION IMPORTANCE ACROSS ENSEMBLE
# ============================================================

def ensemble_permutation_importance(lm, model_dir, K, X_test, y_true, n_repeats=20):
    lm_dir = os.path.join(model_dir, f"LM{lm}")
    importances_list = []

    for i in range(1, K + 1):
        files = [f for f in os.listdir(lm_dir) if f.endswith(f"{i:02d}.joblib")]
        pipe = joblib.load(os.path.join(lm_dir, files[0]))

        result = permutation_importance(
            pipe, X_test, y_true,
            n_repeats=n_repeats,
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1
        )

        importances_list.append(result.importances_mean)

    avg_importances = np.mean(np.vstack(importances_list), axis=0)

    return pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance": avg_importances
    }).sort_values("importance", ascending=False)

feat_importance_df = ensemble_permutation_importance(
    LANDMARK, MODEL_DIR, K, X_test, y_true
)

feat_importance_df.to_csv(
    os.path.join(OUT_DIR, "PermutationImportance_RF_LM18.csv"),
    index=False
)

plt.figure(figsize=(8, 6))
plt.barh(feat_importance_df["feature"], feat_importance_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Permutation Importance (Mean ΔAUROC)")
plt.title("Permutation Feature Importance — RF Ensemble (LM18)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PermutationImportance_RF_LM18.png"), dpi=300)
plt.close()

print("\n✅ Permutation feature importance analysis complete.")
