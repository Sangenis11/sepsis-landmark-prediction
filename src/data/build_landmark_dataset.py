"""
Build a landmark-based long-format dataset for dynamic sepsis prediction.

This script processes patient-level time-series CSV files derived from MIMIC-IV
and constructs landmark observations at 6, 12, 18, and 24 hours after ICU admission.

Each row in the output represents a patient–landmark instance with:
- Features from the preceding 6-hour window
- Outcome: sepsis onset in the subsequent 6 hours
"""

import os
import pandas as pd
from tqdm import tqdm

# ==============================
# 🔹 USER CONFIGURATION
# ==============================

# 👉 CHANGE THIS to the folder containing patient-level CSV files
BASE_PATH = "data/patient_timeseries/"

OUTPUT_FILE = "data/landmark_dataset.csv"

LANDMARKS = [6, 12, 18, 24]   # hours since ICU admission
PREDICTION_WINDOW = 6         # hours after landmark
FEATURE_WINDOW = 6            # hours before landmark

TIME_COL = "hr"
ICU_OUTCOME_COL = "icuouttime_outcome"
DEATH_OUTCOME_COL = "death_outcome"

# ==============================
# FEATURE GROUPS
# ==============================

core_features = [
    'heart_rate', 'mbp', 'spo2', 'temperature',
    'pO2', 'pCO2', 'rr', 'pH',
    'glucose', 'bicarbonate', 'gcs'
]

resp_support = [
    'vasopressor', 'crrt', 'invasive',
    'noninvasive', 'highflow',
    'set_peep', 'set_fio2',
    'set_tv', 'total_tv'
]

bp_vars = ['sbp_ni', 'dbp_ni']
additional_features = ['height_inch', 'pbw_kg']

binary_mode_vars = [
    'vasopressor_mode', 'crrt_mode',
    'invasive_mode', 'noninvasive_mode', 'highflow_mode'
]

demographics = [
    'anchor_age', 'gender',
    'elixhauser_vanwalraven', 'race'
]

all_numeric_features = core_features + resp_support + bp_vars + additional_features

# ==============================
# MAIN PROCESSING FUNCTION
# ==============================

def build_landmark_dataset():
    records = []

    print("🚀 Building landmark dataset...")

    for root, _, files in os.walk(BASE_PATH):
        for file in tqdm(files):
            if not file.endswith(".csv"):
                continue

            try:
                df = pd.read_csv(os.path.join(root, file))

                if 'subject_id' not in df.columns or TIME_COL not in df.columns:
                    continue

                df = df.sort_values(TIME_COL).reset_index(drop=True)
                subject_id = df['subject_id'].iloc[0]

                # ICU exit and death censoring
                icu_exit_time = (
                    df.loc[df[ICU_OUTCOME_COL] == 1, TIME_COL].min()
                    if ICU_OUTCOME_COL in df.columns else float('inf')
                )
                death_time = (
                    df.loc[df[DEATH_OUTCOME_COL] == 1, TIME_COL].min()
                    if DEATH_OUTCOME_COL in df.columns else float('inf')
                )
                risk_end_time = min(
                    icu_exit_time if pd.notna(icu_exit_time) else float('inf'),
                    death_time if pd.notna(death_time) else float('inf')
                )

                for lm in LANDMARKS:

                    # Must be alive and in ICU
                    if lm > risk_end_time:
                        break

                    # Exclude prevalent sepsis
                    if 'sepsis3' in df.columns:
                        if ((df[TIME_COL] < lm) & (df['sepsis3'] == 1)).any():
                            continue

                    feat_df = df[(df[TIME_COL] >= lm - FEATURE_WINDOW) & (df[TIME_COL] < lm)]
                    if feat_df.empty:
                        continue

                    record = {'subject_id': subject_id, 'landmark_hr': lm}

                    # Numeric summaries
                    for var in all_numeric_features:
                        if var in feat_df.columns:
                            vals = pd.to_numeric(feat_df[var], errors='coerce')
                            record[f'{var}_mean'] = vals.mean()
                            record[f'{var}_min'] = vals.min()
                            record[f'{var}_max'] = vals.max()
                        else:
                            record[f'{var}_mean'] = None
                            record[f'{var}_min'] = None
                            record[f'{var}_max'] = None

                    # Binary support modes
                    for bvar in binary_mode_vars:
                        if bvar in feat_df.columns:
                            vals = feat_df[bvar].dropna().astype(int)
                            record[bvar] = int(vals.any()) if not vals.empty else 0
                        else:
                            record[bvar] = 0

                    # BMI
                    h_vals = pd.to_numeric(feat_df['height_inch'], errors='coerce').dropna()
                    w_vals = pd.to_numeric(feat_df['pbw_kg'], errors='coerce').dropna()
                    if not h_vals.empty and not w_vals.empty:
                        h_m = h_vals.iloc[0] * 0.0254
                        record['bmi'] = w_vals.iloc[0] / (h_m ** 2) if h_m > 0 else None
                    else:
                        record['bmi'] = None

                    # Demographics
                    for demo in demographics:
                        record[demo] = df[demo].dropna().iloc[0] if demo in df.columns and not df[demo].dropna().empty else None

                    # Outcome
                    outcome_df = df[(df[TIME_COL] >= lm) & (df[TIME_COL] < lm + PREDICTION_WINDOW)]
                    record['sepsis_next_6h'] = int((outcome_df.get('sepsis3', pd.Series()) == 1).any())

                    records.append(record)

            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")

    output_df = pd.DataFrame(records)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Landmark dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_landmark_dataset()

