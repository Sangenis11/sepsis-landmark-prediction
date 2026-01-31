"""
Interpretability Step: Extract Odds Ratios from Trained Logistic Model (LM18)

This script loads the saved logistic regression pipeline for the 18-hour landmark,
reconstructs the design matrix after preprocessing, and computes:

- Coefficients
- Standard errors (Fisher Information Matrix)
- Odds Ratios (OR)
- 95% Confidence Intervals
- p-values

Output is a CSV table used for manuscript reporting and forest plot creation.
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm

# ==============================
# USER CONFIG
# ==============================
DATA_PATH = "..._READY_noBMI.csv"
PIPELINE_PATH = "outputs/model_results/two_stage/Final_Logistic_LM18.joblib"
OUT_CSV = "outputs/model_results/two_stage/Final_LR_LM18_OddsRatios.csv"

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

OUTCOME = "sepsis_next_6h"

NUMERIC_FEATURES = [
    "heart_rate_max", "spo2_min", "temperature_max",
    "rr_max", "map_ni", "anchor_age"
]

BINARY_FEATURES = [
    "vasopressor_mean", "crrt_mean", "invasive_mean",
    "noninvasive_mean", "highflow_mean",
    "gcs_min_missing", "map_ni_missing"
]

CATEGORICAL_FEATURES = [
    "gender", "race_grouped",
    "gcs_min_cat_num", "elixhauser_cat_num"
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# ==============================
# LOAD DATA (Landmark 18)
# ==============================
df = pd.read_csv(DATA_PATH)
df_lm18 = df[df["landmark_hr"] == 18].copy()

X = df_lm18[ALL_FEATURES]
y = df_lm18[OUTCOME].values

# ==============================
# LOAD TRAINED PIPELINE
# ==============================
pipe = joblib.load(PIPELINE_PATH)
preprocessor = pipe.named_steps["prep"]
model = pipe.named_steps["model"]

# ==============================
# DESIGN MATRIX
# ==============================
X_trans = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

X_design = np.hstack([np.ones((X_trans.shape[0], 1)), X_trans])
feature_names = np.insert(feature_names, 0, "Intercept")

# ==============================
# LINEAR PREDICTOR
# ==============================
beta = np.concatenate([model.intercept_, model.coef_.ravel()])
eta = X_design @ beta
p = 1 / (1 + np.exp(-eta))

# ==============================
# STANDARD ERRORS VIA FISHER INFORMATION
# ==============================
w = p * (1 - p)
Xw = X_design * w[:, None]
XtWX = X_design.T @ Xw
cov = np.linalg.inv(XtWX)
se = np.sqrt(np.diag(cov))

# ==============================
# ODDS RATIOS & CIs
# ==============================
z = norm.ppf(0.975)

OR = np.exp(beta)
CI_lower = np.exp(beta - z * se)
CI_upper = np.exp(beta + z * se)
p_values = 2 * (1 - norm.cdf(np.abs(beta / se)))

results = pd.DataFrame({
    "Variable": feature_names,
    "Coefficient": beta,
    "Std_Error": se,
    "OR": OR,
    "CI_lower": CI_lower,
    "CI_upper": CI_upper,
    "p_value": p_values
})

results = results[results["Variable"] != "Intercept"]
results.to_csv(OUT_CSV, index=False)

print("✅ Odds ratios extracted and saved to:", OUT_CSV)

