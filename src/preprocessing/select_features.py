"""
Preprocessing Step 2: Final feature selection for modeling.

This script selects the clinically and statistically relevant features
used in the prediction models, as defined in the study protocol.
"""

import os
import pandas as pd

# ==============================
# USER CONFIG
# ==============================
INPUT_PATH = "data/landmark_dataset_cleaned.csv"
OUTPUT_PATH = "data/modeling_dataset.csv"

os.makedirs("data", exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_PATH)

# ==============================
# SELECTED FEATURES
# ==============================
selected_columns = [
    # Identifiers & time anchor
    "subject_id",
    "landmark_hr",

    # Physiological extremes
    "heart_rate_max",
    "spo2_min",
    "temperature_max",
    "rr_max",
    "gcs_min",

    # Blood pressure
    "sbp_ni_mean",
    "dbp_ni_mean",

    # Clinical interventions
    "vasopressor_mean",
    "crrt_mean",
    "invasive_mean",
    "noninvasive_mean",
    "highflow_mean",

    # Static characteristics
    "anchor_age",
    "gender",
    "bmi",
    "elixhauser_vanwalraven",
    "race",

    # Outcome
    "sepsis_next_6h"
]

# ==============================
# SAFETY CHECK
# ==============================
missing_cols = set(selected_columns) - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# ==============================
# SUBSET DATA
# ==============================
df_filtered = df[selected_columns]

# ==============================
# SAVE OUTPUT
# ==============================
df_filtered.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Modeling dataset saved to: {OUTPUT_PATH}")
print(f"📊 Final shape: {df_filtered.shape}")

