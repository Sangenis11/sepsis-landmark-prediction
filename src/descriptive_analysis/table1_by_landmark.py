"""
==============================================================
Landmark-Specific Table 1 Generator
--------------------------------------------------------------
This script creates one comparative descriptive table per
landmark hour (6h, 12h, 18h, 24h) stratified by sepsis outcome.

Outputs:
    CSV files saved in: output/table1_by_landmark/

Input required:
    Place the dataset file:
    data/..._READY_noBMI.csv

Author: Your Name
Project: Dynamic Landmark-Based Sepsis Prediction
==============================================================
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

# ================================
# FILE PATHS (RELATIVE FOR GITHUB)
# ================================
file_path = "../data/..._READY_noBMI.csv"
out_dir = "../output/table1_by_landmark"
os.makedirs(out_dir, exist_ok=True)

landmarks = [6, 12, 18, 24]
outcome_col = "sepsis_next_6h"

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(file_path)
print("Dataset loaded:", df.shape)

# ================================
# HUMAN-READABLE LABELS
# ================================
gcs_labels = {1: "Severe", 2: "Moderate", 3: "Mild"}
elix_labels = {1: "Low", 2: "Moderate", 3: "High"}

df["gcs_min_cat"] = df["gcs_min_cat_num"].map(gcs_labels)
df["elixhauser_cat"] = df["elixhauser_cat_num"].map(elix_labels)

# ================================
# VARIABLE GROUPS
# ================================
continuous_vars = [
    "anchor_age", "heart_rate_max", "spo2_min",
    "temperature_max", "rr_max", "map_ni"
]

categorical_vars = [
    "gender", "race_grouped", "vasopressor_mean", "crrt_mean",
    "invasive_mean", "noninvasive_mean", "highflow_mean",
    "gcs_min_cat", "elixhauser_cat", "hypotension", "gcs_severe"
]

# ================================
# HELPER FUNCTIONS
# ================================
def median_iqr(series):
    return f"{series.median():.1f} ({series.quantile(0.25):.1f}–{series.quantile(0.75):.1f})"

def n_pct(count, total):
    return f"{count} ({count / total * 100:.1f}%)" if total > 0 else "0 (0%)"

def mw_pvalue(a, b):
    if len(a.dropna()) > 0 and len(b.dropna()) > 0:
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    return np.nan

def cat_pvalue(table):
    if table.shape == (2, 2):
        return fisher_exact(table)[1]
    else:
        return chi2_contingency(table)[1]

# ================================
# GENERATE TABLES PER LANDMARK
# ================================
for lm in landmarks:

    lm_df = df[df["landmark_hr"] == lm].copy()
    no_sepsis = lm_df[lm_df[outcome_col] == 0]
    sepsis = lm_df[lm_df[outcome_col] == 1]

    rows = []

    # ---------- CONTINUOUS ----------
    for var in continuous_vars:
        row = {"Variable": var}
        row["No Sepsis"] = median_iqr(no_sepsis[var].dropna())
        row["Sepsis"] = median_iqr(sepsis[var].dropna())
        row["p-value"] = mw_pvalue(no_sepsis[var], sepsis[var])
        rows.append(row)

    # ---------- CATEGORICAL ----------
    for var in categorical_vars:
        levels = sorted(lm_df[var].dropna().unique())

        for level in levels:
            row = {"Variable": f"{var}: {level}"}

            n0 = (no_sepsis[var] == level).sum()
            n1 = (sepsis[var] == level).sum()

            row["No Sepsis"] = n_pct(n0, len(no_sepsis))
            row["Sepsis"] = n_pct(n1, len(sepsis))

            contingency = pd.crosstab(lm_df[var] == level, lm_df[outcome_col])
            row["p-value"] = cat_pvalue(contingency) if contingency.shape[1] == 2 else np.nan

            rows.append(row)

    # ---------- MISSINGNESS ----------
    for var in continuous_vars:
        row = {"Variable": f"{var} missing"}
        row["No Sepsis"] = n_pct(no_sepsis[var].isna().sum(), len(no_sepsis))
        row["Sepsis"] = n_pct(sepsis[var].isna().sum(), len(sepsis))
        row["p-value"] = np.nan
        rows.append(row)

    # Save table
    table1 = pd.DataFrame(rows)
    table1.insert(1, "Landmark", f"{lm}h")

    out_file = os.path.join(out_dir, f"Table1_Landmark_{lm}h.csv")
    table1.to_csv(out_file, index=False)

    print(f"Table saved for {lm}h → {out_file}")

print("\nAll landmark-specific tables generated successfully.")

