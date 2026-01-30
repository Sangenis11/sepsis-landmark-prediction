"""
Preprocessing Step 5: Clinical feature engineering.

This script performs:
1. Binarization of organ support exposure variables
2. Categorization of Glasgow Coma Scale (GCS)
3. Categorization of Elixhauser comorbidity score
"""

import os
import pandas as pd
import numpy as np

# ==============================
# USER CONFIG
# ==============================
INPUT_PATH = "data/modeling_dataset_encoded.csv"
OUTPUT_PATH = "data/modeling_dataset_final.csv"

os.makedirs("data", exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_PATH)

# =====================================================
# 1️⃣ BINARIZE INTERVENTION EXPOSURE (≥50% WINDOW RULE)
# =====================================================
binary_avg_cols = [
    "vasopressor_mean",
    "crrt_mean",
    "invasive_mean",
    "noninvasive_mean",
    "highflow_mean"
]

df[binary_avg_cols] = df[binary_avg_cols].apply(pd.to_numeric, errors="coerce")

for col in binary_avg_cols:
    df[col] = np.where(df[col].isna(), np.nan,
                       np.where(df[col] >= 0.5, 1, 0))

print("✅ Intervention variables binarized")

# =====================================
# 2️⃣ GCS SEVERITY CATEGORIZATION
# =====================================
df['gcs_min_cat'] = pd.cut(
    df['gcs_min'],
    bins=[-1, 8, 12, 15],
    labels=['Severe', 'Moderate', 'Mild'],
    include_lowest=True
)

gcs_mapping = {'Severe': 1, 'Moderate': 2, 'Mild': 3}
df['gcs_min_cat_num'] = df['gcs_min_cat'].map(gcs_mapping)

df['gcs_severe'] = np.where(df['gcs_min'].isna(), np.nan,
                            (df['gcs_min'] <= 8).astype(int))

print("✅ GCS categories created")

# =====================================
# 3️⃣ ELIXHAUSER COMORBIDITY CATEGORIES
# =====================================
def categorize_elixhauser(score):
    if pd.isna(score):
        return "Missing"
    elif score <= 4:
        return "Low comorbidities"
    elif score <= 14:
        return "Moderate comorbidities"
    else:
        return "High comorbidities"

df["elixhauser_cat"] = df["elixhauser_vanwalraven"].apply(categorize_elixhauser)

majority_class = (
    df.loc[df["elixhauser_cat"] != "Missing", "elixhauser_cat"]
    .value_counts()
    .idxmax()
)

df["elixhauser_cat"] = df["elixhauser_cat"].replace("Missing", majority_class)

elixhauser_mapping = {
    "Low comorbidities": 1,
    "Moderate comorbidities": 2,
    "High comorbidities": 3
}
df["elixhauser_cat_num"] = df["elixhauser_cat"].map(elixhauser_mapping)

print("✅ Elixhauser categories created")

# ==============================
# SAVE OUTPUT
# ==============================
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n🎯 Final modeling dataset saved to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")

