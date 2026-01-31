"""
Evaluation Step: Calibration Analysis for Best Models at Each Landmark

This script evaluates probability calibration on the held-out test set for the
best-performing model at each landmark (6h, 12h, 18h, 24h).

It produces:
- Calibration curves (PNG)
- Calibration tables (CSV)
- Brier scores summary (CSV)

These outputs are used for model reliability assessment in the manuscript.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ==============================
# USER CONFIG
# ==============================
DATA_PATH = "data/processed/V02_cohort_ANALYSIS_READY_noBMI.csv"
MODEL_DIR = "outputs/model_results/two_stage"
OUT_DIR   = os.path.join(MODEL_DIR, "calibration")

os.makedirs(OUT_DIR, exist_ok=True)

LANDMARKS = [6, 12, 18, 24]

BEST_MODELS = {
    6: "RandomForest",
    12: "Logistic",
    18: "Logistic",
    24: "Logistic"
}

OUTCOME = "sepsis_next_6h"
PATIENT_ID = "subject_id"
N_BINS = 10

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded:", df.shape)

# ==============================
# LOAD TRAIN / TEST SPLIT
# ==============================
train_patients = pd.read_csv(
    os.path.join(MODEL_DIR, "train_patients.csv"),
    header=None
)[0].tolist()

test_patients = pd.read_csv(
    os.path.join(MODEL_DIR, "test_patients.csv"),
    header=None
)[0].tolist()

df_test = df[df[PATIENT_ID].isin(test_patients)]
print("✅ Test set rows:", df_test.shape)

# ==============================
# CALIBRATION ANALYSIS
# ==============================
calibration_summary = []

for lm in LANDMARKS:
    model_name = BEST_MODELS[lm]
    print(f"\n🔹 Calibration — Landmark {lm}h ({model_name})")

    model_path = os.path.join(
        MODEL_DIR,
        f"Final_{model_name}_LM{lm}.joblib"
    )

    pipe = joblib.load(model_path)

    df_lm = df_test[df_test["landmark_hr"] == lm]

    if df_lm.empty:
        print("⚠️ No test data at this landmark — skipped")
        continue

    X_test = df_lm.drop(columns=[OUTCOME])
    y_test = df_lm[OUTCOME].values

    y_prob = pipe.predict_proba(X_test)[:, 1]

    # ------------------------------
    # Calibration curve
    # ------------------------------
    prob_true, prob_pred = calibration_curve(
        y_test,
        y_prob,
        n_bins=N_BINS,
        strategy="quantile"
    )

    # ------------------------------
    # Brier score
    # ------------------------------
    brier = brier_score_loss(y_test, y_prob)

    calibration_summary.append({
        "landmark_hr": lm,
        "model": model_name,
        "Brier_score": brier,
        "n_samples": len(y_test),
        "n_events": int(y_test.sum())
    })

    # ------------------------------
    # Calibration plot
    # ------------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Event Frequency")
    plt.title(f"Calibration Curve — LM {lm}h ({model_name})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"Calibration_Curve_LM{lm}_{model_name}.png"),
        dpi=300
    )
    plt.close()

    # ------------------------------
    # Calibration table
    # ------------------------------
    calib_table = pd.DataFrame({
        "Bin": np.arange(1, len(prob_pred) + 1),
        "Predicted_prob": prob_pred,
        "Observed_freq": prob_true
    })

    calib_table.to_csv(
        os.path.join(OUT_DIR, f"Calibration_Table_LM{lm}_{model_name}.csv"),
        index=False
    )

    print(f"   ✅ Brier score: {brier:.4f}")

# ==============================
# SAVE SUMMARY
# ==============================
summary_df = pd.DataFrame(calibration_summary)
summary_df.to_csv(
    os.path.join(OUT_DIR, "Calibration_Summary.csv"),
    index=False
)

print("\n✅ Calibration analysis completed successfully")
print(summary_df)

