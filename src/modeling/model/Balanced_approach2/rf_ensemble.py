"""
Modeling Approach 2C: Balanced Random Forest Ensemble (Under-Sampling)

This script implements an EasyEnsemble-style strategy using Random Forests
for landmark-based sepsis prediction under severe class imbalance.

For each landmark:
• All positive cases are retained
• Negative cases are divided into K subsets
• One Random Forest is trained per balanced subset
• Predictions are averaged across ensemble members

This avoids synthetic sampling and improves robustness under imbalance.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.utils import shuffle

# ==============================
# USER CONFIG
# ==============================
DATA_PATH = "data/processed/V02_cohort_ANALYSIS_READY_noBMI.csv"
OUT_DIR   = "outputs/model_results/balanced_rf_ensemble"
os.makedirs(OUT_DIR, exist_ok=True)

OUTCOME = "sepsis_next_6h"
PATIENT_ID = "subject_id"
LANDMARKS = [6, 12, 18, 24]
RANDOM_STATE = 42

K_DICT = {
    6: 20,
    12: 30,
    18: 40,
    24: 50
}

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded:", df.shape)

# ==============================
# FEATURE GROUPS
# ==============================
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

# ==============================
# PREPROCESSING (no scaling needed for RF)
# ==============================
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ("num", "passthrough", NUMERIC_FEATURES + BINARY_FEATURES)
])

# ==============================
# PATIENT-LEVEL SPLIT
# ==============================
patients = df[PATIENT_ID].unique()
train_pats, test_pats = train_test_split(patients, test_size=0.25, random_state=RANDOM_STATE)

df_train = df[df[PATIENT_ID].isin(train_pats)]
df_test  = df[df[PATIENT_ID].isin(test_pats)]

# ==============================
# PLOTTING HELPERS
# ==============================
def plot_roc(y_true, y_prob, lm):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – RF Balanced Ensemble (LM {lm}h)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ROC_RF_BE_LM{lm}.png"), dpi=300)
    plt.close()

def plot_pr(y_true, y_prob, lm):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(r, p, label=f"AUPRC = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve – RF Balanced Ensemble (LM {lm}h)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"PR_RF_BE_LM{lm}.png"), dpi=300)
    plt.close()

# ==============================
# BALANCED RF ENSEMBLE FUNCTION
# ==============================
def balanced_rf_ensemble(df_train_lm, df_test_lm, lm, K):

    lm_dir = os.path.join(OUT_DIR, f"LM{lm}")
    os.makedirs(lm_dir, exist_ok=True)

    df_pos = df_train_lm[df_train_lm[OUTCOME] == 1]
    df_neg = df_train_lm[df_train_lm[OUTCOME] == 0]
    df_neg = shuffle(df_neg, random_state=RANDOM_STATE)

    neg_chunks = np.array_split(df_neg, K)
    test_probs = []

    for i, neg_k in enumerate(neg_chunks, start=1):

        df_bal = pd.concat([df_pos, neg_k])

        X_train = df_bal[ALL_FEATURES]
        y_train = df_bal[OUTCOME]

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=50,
                class_weight="balanced",
                n_jobs=-1,
                random_state=i
            ))
        ])

        pipe.fit(X_train, y_train)

        joblib.dump(pipe, os.path.join(lm_dir, f"rf_model_{i:02d}.joblib"))

        y_prob = pipe.predict_proba(df_test_lm[ALL_FEATURES])[:, 1]
        test_probs.append(y_prob)

    return np.mean(np.vstack(test_probs), axis=0)

# ==============================
# RUN ACROSS LANDMARKS
# ==============================
results = []

for lm in LANDMARKS:
    print(f"\n🔹 Running RF Balanced Ensemble — Landmark {lm}h")

    df_train_lm = df_train[df_train["landmark_hr"] == lm]
    df_test_lm  = df_test[df_test["landmark_hr"] == lm]

    y_test = df_test_lm[OUTCOME]
    K = K_DICT[lm]

    y_prob = balanced_rf_ensemble(df_train_lm, df_test_lm, lm, K)

    plot_roc(y_test, y_prob, lm)
    plot_pr(y_test, y_prob, lm)

    results.append({
        "landmark_hr": lm,
        "model": f"BalancedRF_Ensemble_K{K}",
        "AUROC": roc_auc_score(y_test, y_prob),
        "AUPRC": average_precision_score(y_test, y_prob),
        "n_samples": len(y_test),
        "n_events": int(y_test.sum())
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "Final_Test_BalancedRF_Ensemble.csv"), index=False)

print("\n✅ RF Balanced Ensemble evaluation complete")
print(results_df)

