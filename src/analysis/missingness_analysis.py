"""
Compute and visualize variable missingness by landmark hour.

This script summarizes the percentage missingness for time-varying features
across landmark time points in the constructed landmark dataset.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# 🔹 USER CONFIGURATION
# ==============================

DATA_PATH = "data/landmark_dataset.csv"
OUTPUT_DIR = "results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)

if 'landmark_hr' not in df.columns:
    raise ValueError("Missing required column 'landmark_hr' in the dataset")

print("Dataset shape:", df.shape)
print("Landmarks:", sorted(df['landmark_hr'].unique()))

# ==============================
# SELECT VARIABLES FOR ANALYSIS
# ==============================
exclude_cols = [
    'subject_id',
    'landmark_hr',
    'anchor_age',
    'gender',
    'race',
    'bmi',
    'sepsis_next_6h'
]

analysis_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Variables used for missingness analysis: {len(analysis_cols)}")

# ==============================
# COMPUTE MISSINGNESS
# ==============================
missing_by_landmark = (
    df.groupby('landmark_hr')[analysis_cols]
      .apply(lambda x: x.isnull().mean() * 100)
      .T
)

# ==============================
# SAVE CSV OUTPUT
# ==============================
csv_output_path = os.path.join(OUTPUT_DIR, "missingness_by_landmark.csv")
missing_by_landmark.to_csv(csv_output_path)

print("✅ Missingness summary saved to:", csv_output_path)

# ==============================
# SAVE HEATMAP
# ==============================
plt.figure(figsize=(14, 10))
sns.heatmap(missing_by_landmark, cmap='OrRd', annot=False)
plt.title("Missingness (%) per Variable by Landmark Hour")
plt.xlabel("Landmark Hour")
plt.ylabel("Variable")
plt.tight_layout()

fig_output_path = os.path.join(OUTPUT_DIR, "missingness_heatmap.png")
plt.savefig(fig_output_path, dpi=300)
plt.close()

print("📊 Heatmap saved to:", fig_output_path)

