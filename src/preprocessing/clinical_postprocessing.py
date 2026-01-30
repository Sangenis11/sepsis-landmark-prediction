"""
Preprocessing Step 6: Clinical post-processing.

This script:
1. Derives Mean Arterial Pressure (MAP) and hypotension indicator
2. Applies clinically informed outlier capping
3. Ensures missingness indicators are consistent after feature derivation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# USER CONFIG
# ==============================
INPUT_PATH = "data/modeling_dataset_final.csv"
OUTPUT_PATH = "data/modeling_dataset_ready.csv"
PLOT_DIR = "results/capping_plots/"

os.makedirs("data", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_PATH)
print("Initial shape:", df.shape)

# =====================================================
# 1️⃣ DERIVE MAP AND HYPOTENSION
# =====================================================
df["sbp_ni_mean"] = pd.to_numeric(df["sbp_ni_mean"], errors="coerce")
df["dbp_ni_mean"] = pd.to_numeric(df["dbp_ni_mean"], errors="coerce")

df["map_ni"] = (df["sbp_ni_mean"] + 2 * df["dbp_ni_mean"]) / 3

df["hypotension"] = np.where(
    df["map_ni"].isna(),
    np.nan,
    (df["map_ni"] < 65).astype(int)
)

df.drop(columns=["sbp_ni_mean", "dbp_ni_mean"], inplace=True)
print("✅ MAP and hypotension derived")

# =====================================================
# 2️⃣ CLINICAL OUTLIER CAPPING
# =====================================================
medical_ranges = {
    "heart_rate_max": (40, 180),
    "spo2_min": (50, 100),
    "temperature_max": (35.0, 40.5),
    "rr_max": (8, 35),
    "anchor_age": (18, 120),
    "bmi": (10, 60),
    "map_ni": (40, 120)
}

df_before = df.copy()
summary_rows = []

for var, (low, high) in medical_ranges.items():
    if var not in df.columns:
        continue

    orig = df[var]
    below = (orig < low).sum()
    above = (orig > high).sum()

    df[var] = orig.clip(lower=low, upper=high)

    summary_rows.append({
        "Variable": var,
        "Lower": low,
        "Upper": high,
        "Below": below,
        "Above": above,
        "Percent_Capped": round(100 * (below + above) / orig.notna().sum(), 2)
    })

    plt.figure(figsize=(6,4))
    plt.boxplot([df_before[var].dropna(), df[var].dropna()], labels=["Before","After"])
    plt.title(f"{var} Before vs After Capping")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{var}_capping.png"))
    plt.close()

pd.DataFrame(summary_rows).to_csv("results/capping_summary.csv", index=False)
print("✅ Clinical capping applied")

# =====================================================
# 3️⃣ ENSURE MISSINGNESS INDICATORS
# =====================================================
base_missing_vars = [
    "heart_rate_max", "spo2_min", "temperature_max",
    "rr_max", "gcs_min", "bmi"
]

for var in base_missing_vars:
    col = f"{var}_missing"
    if col not in df.columns and var in df.columns:
        df[col] = df[var].isna().astype(int)

df["map_ni_missing"] = df["map_ni"].isna().astype(int)
print("✅ Missingness indicators updated")

# ==============================
# SAVE OUTPUT
# ==============================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n🎯 Final modeling-ready dataset saved to: {OUTPUT_PATH}")
print("Final shape:", df.shape)

