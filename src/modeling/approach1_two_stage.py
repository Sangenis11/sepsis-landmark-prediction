"""
Modeling Approach 1: Standard Two-Stage Machine Learning Pipeline

This script implements the primary modeling framework described in the study:

Stage 1:
- Patient-level train/test split
- Grouped cross-validation within the training set
- Model comparison (Logistic Regression, Random Forest, XGBoost)

Stage 2:
- Final model training on full training data
- Independent evaluation on held-out test data

Models are trained separately at each landmark hour (6, 12, 18, 24).
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
# USER CONFIG
# ==============================
DATA_PATH = "data/processed/V02_cohort_ANALYSIS_READY_noBMI.csv"
OUT_DIR = "outputs/model_results/two_stage"

os.makedirs(OUT_DIR, exist_ok=True)

LANDMARKS = [6, 12, 18, 24]
N_SPLITS = 5
RANDOM_STATE = 42

OUTCOME = "sepsis_next_6h"
PATIENT_ID = "subject_id"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded:", df.shape)

# ==============================
# FEATURE GROUPS
# ==============================
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
# PREPROCESSING PIPELINE
# ==============================
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("bin", "passthrough", BINARY_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ==============================
# STAGE 1 — CROSS-VALIDATION
# ==============================
def stage1_cv(df):

    unique_patients = df[PATIENT_ID].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.25, random_state=RANDOM_STATE
    )

    df_train = df[df[PATIENT_ID].isin(train_patients)]
    df_test  = df[df[PATIENT_ID].isin(test_patients)]

    models_dict = {
        "Logistic": LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                               class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1)
    }

    cv_results = []

    for lm in LANDMARKS:
        df_lm = df_train[df_train["landmark_hr"] == lm]
        X = df_lm[ALL_FEATURES]
        y = df_lm[OUTCOME]
        groups = df_lm[PATIENT_ID]

        gkf = GroupKFold(n_splits=N_SPLITS)

        for model_name, model in models_dict.items():

            aurocs, auprcs, sens, spec = [], [], [], []

            for tr, va in gkf.split(X, y, groups):

                pipe = Pipeline([("prep", preprocessor), ("model", model)])
                pipe.fit(X.iloc[tr], y.iloc[tr])

                y_prob = pipe.predict_proba(X.iloc[va])[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                tn, fp, fn, tp = confusion_matrix(y.iloc[va], y_pred).ravel()

                aurocs.append(roc_auc_score(y.iloc[va], y_prob))
                auprcs.append(average_precision_score(y.iloc[va], y_prob))
                sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

            cv_results.append({
                "landmark_hr": lm,
                "model": model_name,
                "AUROC_mean": np.mean(aurocs),
                "AUPRC_mean": np.mean(auprcs),
                "Sensitivity_mean": np.mean(sens),
                "Specificity_mean": np.mean(spec),
                "n_patients": df_lm[PATIENT_ID].nunique(),
                "n_events": int(y.sum())
            })

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(OUT_DIR, "CV_Performance.csv"), index=False)

    print("✅ Stage 1 CV complete")
    return df_train, df_test

# ==============================
# STAGE 2 — FINAL TEST EVALUATION
# ==============================
def stage2_final_eval(df_train, df_test, best_models):

    final_results = []

    for lm in LANDMARKS:
        df_train_lm = df_train[df_train["landmark_hr"] == lm]
        df_test_lm  = df_test[df_test["landmark_hr"] == lm]

        X_train = df_train_lm[ALL_FEATURES]
        y_train = df_train_lm[OUTCOME]
        X_test  = df_test_lm[ALL_FEATURES]
        y_test  = df_test_lm[OUTCOME]

        model_name = best_models[lm]

        model = (
            LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000, class_weight="balanced")
            if model_name == "Logistic" else
            RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                   class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
            if model_name == "RandomForest" else
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1)
        )

        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        final_results.append({
            "landmark_hr": lm,
            "model": model_name,
            "AUROC": roc_auc_score(y_test, y_prob),
            "AUPRC": average_precision_score(y_test, y_prob),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1_score": f1_score(y_test, y_pred, zero_division=0)
        })

        joblib.dump(pipe, os.path.join(OUT_DIR, f"Final_{model_name}_LM{lm}.joblib"))

    final_df = pd.DataFrame(final_results)
    final_df.to_csv(os.path.join(OUT_DIR, "Final_Test_Performance.csv"), index=False)

    print("✅ Stage 2 final evaluation complete")
    return final_df

# ==============================
# RUN PIPELINE
# ==============================
df_train, df_test = stage1_cv(df)

best_models = {6: "RandomForest", 12: "Logistic", 18: "Logistic", 24: "Logistic"}

final_results = stage2_final_eval(df_train, df_test, best_models)

print("\n📄 Final Test Results:")
print(final_results)

