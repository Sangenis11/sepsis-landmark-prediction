"""
Interpretability Step: Forest Plot for Logistic Model (LM18)

This script creates a publication-ready forest plot of adjusted odds ratios
derived from the logistic regression model at the 18-hour landmark.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# USER CONFIG
# ==============================
IN_CSV = "outputs/model_results/two_stage/Final_LR_LM18_OddsRatios.csv"
OUT_PNG = "outputs/model_results/two_stage/Final_LR_LM18_ForestPlot.png"

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df_plot = pd.read_csv(IN_CSV)

# Remove invalid estimates
df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
df_plot = df_plot.dropna(subset=["OR", "CI_lower", "CI_upper"])
df_plot = df_plot[df_plot["CI_upper"] < 1e6]

# ==============================
# CLEAN FEATURE LABELS
# ==============================
LABEL_MAP = {
    "heart_rate_max": "Maximum Heart Rate",
    "spo2_min": "Minimum SpO₂",
    "temperature_max": "Maximum Temperature",
    "rr_max": "Maximum Respiratory Rate",
    "map_ni": "MAP (Non-invasive)",
    "anchor_age": "Age",
    "vasopressor_mean": "Vasopressor Use",
    "crrt_mean": "CRRT Use",
    "invasive_mean": "Invasive Ventilation",
    "noninvasive_mean": "Non-invasive Ventilation",
    "highflow_mean": "High-flow Oxygen",
    "gcs_min_missing": "GCS Missing Indicator",
    "map_ni_missing": "MAP Missing Indicator",
    "gender": "Gender",
    "race_grouped": "Race",
    "gcs_min_cat_num": "GCS (ordinal)",
    "elixhauser_cat_num": "Elixhauser Category"
}

def clean_name(x):
    x = x.replace("num__", "").replace("bin__", "").replace("cat__", "")
    return LABEL_MAP.get(x, x.replace("_", " ").title())

df_plot["Label"] = df_plot["Variable"].apply(clean_name)
df_plot = df_plot.sort_values("OR")
y_pos = np.arange(len(df_plot))

# ==============================
# PLOT
# ==============================
fig, ax = plt.subplots(figsize=(12, 0.35 * len(df_plot)))

ax.errorbar(
    df_plot["OR"],
    y_pos,
    xerr=[df_plot["OR"] - df_plot["CI_lower"], df_plot["CI_upper"] - df_plot["OR"]],
    fmt="o",
    color="black",
    ecolor="black",
    elinewidth=1.4,
    capsize=3
)

ax.axvline(1, linestyle="--", linewidth=1, color="red")

ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot["Label"])
ax.set_xscale("log")

ax.set_xlabel("Odds Ratio (log scale)")
ax.set_title("Adjusted Odds Ratios — Logistic Model at 18h Landmark")

ax.grid(True, axis="x", linestyle=":", linewidth=0.7)
ax.grid(False, axis="y")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.show()

print("✅ Forest plot saved to:", OUT_PNG)

