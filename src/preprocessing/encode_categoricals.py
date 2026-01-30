"""
Preprocessing Step 4: Harmonize categorical variables.

This script standardizes and groups race/ethnicity categories into
clinically meaningful and statistically usable groups.
"""

import os
import pandas as pd

# ==============================
# USER CONFIG
# ==============================
INPUT_PATH = "data/modeling_dataset_imputed.csv"
OUTPUT_PATH = "data/modeling_dataset_encoded.csv"

os.makedirs("data", exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_PATH)

# ==============================
# STANDARDIZE RACE LABELS
# ==============================
df['race'] = df['race'].astype(str).str.upper().str.strip()

race_mapping = {
    # White
    'WHITE': 'White',
    'WHITE - OTHER EUROPEAN': 'White',
    'WHITE - RUSSIAN': 'White',
    'WHITE - EASTERN EUROPEAN': 'White',
    'WHITE - BRAZILIAN': 'White',

    # Black
    'BLACK/AFRICAN AMERICAN': 'Black',
    'BLACK/CAPE VERDEAN': 'Black',
    'BLACK/CARIBBEAN ISLAND': 'Black',
    'BLACK/AFRICAN': 'Black',

    # Asian
    'ASIAN': 'Asian',
    'ASIAN - CHINESE': 'Asian',
    'ASIAN - SOUTH EAST ASIAN': 'Asian',
    'ASIAN - ASIAN INDIAN': 'Asian',
    'ASIAN - KOREAN': 'Asian',

    # Hispanic/Latino
    'HISPANIC OR LATINO': 'Hispanic/Latino',
    'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - DOMINICAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - GUATEMALAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - SALVADORAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - MEXICAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - CUBAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - COLUMBIAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - HONDURAN': 'Hispanic/Latino',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic/Latino',
    'SOUTH AMERICAN': 'Hispanic/Latino',
    'PORTUGUESE': 'Hispanic/Latino',

    # Other groups
    'AMERICAN INDIAN/ALASKA NATIVE': 'Native American',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Pacific Islander',

    # Unknown / other
    'UNKNOWN': 'Other/Unknown',
    'UNABLE TO OBTAIN': 'Other/Unknown',
    'PATIENT DECLINED TO ANSWER': 'Other/Unknown',
    'MULTIPLE RACE/ETHNICITY': 'Other/Unknown',
    'OTHER': 'Other/Unknown'
}

df['race_grouped'] = df['race'].map(race_mapping).fillna('Other/Unknown')

print("Race category counts:")
print(df['race_grouped'].value_counts())

# ==============================
# SAVE OUTPUT
# ==============================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Encoded dataset saved to: {OUTPUT_PATH}")

