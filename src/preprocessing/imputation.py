"""
Preprocessing Step 3: Missing data imputation and indicator creation.

This script performs:
1. Creation of missingness indicator variables
2. Landmark-aware median imputation for time-varying features
3. Global median imputation for static features
"""

import os
import pandas as pd

# ==============================
# USER CONFIG
# ==============================
INPUT_PATH = "data/modeling_dataset.csv"
OUTPUT_PATH = "data/modeling_dataset_imputed.csv"

os.makedirs("data", exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_PATH)
print("Original shape:", df.shape)

# ==============================
# VARIABLE GROUPS
# ==============================
landmark_numeric_vars = [
    "heart_rate_max",
    "spo2_min",
    "temperature_max",
    "rr_max",
    "gcs_min",
    "sbp_ni_mean",
    "dbp_ni_mean"
]

static_numeric_vars = ["bmi"]

indicator_vars = [
    "gcs_min",
    "sbp_ni_mean",
    "dbp_ni_mean",
    "bmi"
]

# ==============================
# CREATE MISSING INDICATORS
# ==============================
for var in indicator_vars:
    if var in df.columns:
        df[f"{var}_missing"] = df[var].isnull().astype(int)

print("✅ Missing indicators created")

# ==============================
# LANDMARK-AWARE MEDIAN IMPUTATION
# ==============================
for var in landmark_numeric_vars:
    if var in df.columns:
        df[var] = df.groupby("landmark_hr")[var].transform(lambda x: x.fillna(x.median()))

print("✅ Landmark-aware median imputation completed")

# ==============================
# GLOBAL MEDIAN IMPUTATION (STATIC)
# ==============================
for var in static_numeric_vars:
    if var in df.columns:
        df[var] = df[var].fillna(df[var].median())

print("✅ Global median imputation completed")

# ==============================
# SAFETY CHECK
# ==============================
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if remaining_missing.empty:
    print("🎉 No missing values remain (except categorical if present)")
else:
    print("⚠️ Remaining missing values detected:")
    print(remaining_missing)

# ==============================
# SAVE OUTPUT
# ==============================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Imputed dataset saved to: {OUTPUT_PATH}")
print("Final shape:", df.shape)

