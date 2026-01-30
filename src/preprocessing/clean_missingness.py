"""
Preprocessing Step 1: Missingness-based feature and row filtering.

This script:
1. Removes variables with >60% missingness
2. Removes redundant support-mode and duplicate summary variables
3. Removes rows with missing outcome or landmark
4. Removes rows with no clinical information
5. Saves cleaned dataset for imputation
"""

import os
import pandas as pd

# ==============================
# USER CONFIG
# ==============================
INPUT_DATA = "data/landmark_dataset.csv"
MISSINGNESS_TABLE = "results/missingness_by_landmark.csv"
OUTPUT_PATH = "data/landmark_dataset_cleaned.csv"

os.makedirs("data", exist_ok=True)

# ==============================
# 1️⃣ DROP HIGH-MISSING FEATURES
# ==============================
df = pd.read_csv(INPUT_DATA)
missing_summary = pd.read_csv(MISSINGNESS_TABLE, index_col=0)

to_drop = missing_summary[missing_summary.max(axis=1) > 60].index.tolist()
print(f"📉 Dropping {len(to_drop)} columns with >60% missingness")

df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

# ==============================
# 2️⃣ DROP REDUNDANT VARIABLES
# ==============================
cols_to_drop = [
    'vasopressor_mode','crrt_mode','invasive_mode','noninvasive_mode','highflow_mode',
    'height_inch_mean','height_inch_min','height_inch_max',
    'pbw_kg_mean','pbw_kg_min','pbw_kg_max',
    'vasopressor_min','vasopressor_max',
    'crrt_min','crrt_max',
    'invasive_min','invasive_max',
    'noninvasive_min','noninvasive_max',
    'highflow_min','highflow_max'
]

df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
print("🧹 Removed redundant support/duplicate variables")

# ==============================
# 3️⃣ REMOVE INVALID ROWS
# ==============================
df = df.dropna(subset=["sepsis_next_6h", "landmark_hr"])
print("🚮 Dropped rows with missing outcome or landmark")

clinically_informative_cols = [
    "heart_rate_max", "sbp_ni_min", "dbp_ni_min",
    "spo2_min", "rr_min", "temperature_max", "vasopressor_mean"
]

df = df[~df[clinically_informative_cols].isnull().all(axis=1)]
print("🩺 Dropped clinically empty rows")

# ==============================
# 4️⃣ SAVE OUTPUT
# ==============================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Cleaned dataset saved to: {OUTPUT_PATH}")
print(f"📊 Final shape: {df.shape}")

