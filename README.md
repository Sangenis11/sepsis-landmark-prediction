🏥 Landmark-Based Early Prediction of Sepsis in ICU Patients
Interpretable and Imbalance-Aware Machine Learning Framework

This repository contains the full analysis pipeline for dynamic landmark-based prediction of sepsis in respiratory-supported critically ill patients. The framework integrates clinically interpretable modeling, class imbalance handling, and transparent evaluation aligned with reproducible research principles.

📌 Study Overview

Sepsis remains a leading cause of mortality in the ICU. Early detection is crucial but challenging due to:

Rapid physiological deterioration

Highly imbalanced outcome distribution

Time-dependent clinical progression

This project implements a landmark-based prediction strategy, generating sepsis risk predictions at:

6h, 12h, 18h, and 24h after ICU admission.

Each landmark uses only information available up to that time point, mimicking real-world clinical deployment.

🧠 Modeling Approaches

We implemented two complementary modeling strategies:

1️⃣ Standard Two-Stage Modeling

Patient-level split → Cross-validation → Independent test evaluation

Models:

Logistic Regression (interpretable baseline)

Random Forest

XGBoost

Class imbalance handled using:

Class weights

Performance metrics robust to imbalance (AUROC, AUPRC)

2️⃣ Balanced Ensemble (EasyEnsemble-style)

To better address extreme imbalance, we built balanced sub-model ensembles:

Majority class split into multiple subsets

Each sub-model trained on all positives + one subset of negatives

Final prediction = average probability across sub-models

Ensemble models:

Logistic Regression Ensemble

Random Forest Ensemble

XGBoost Ensemble

This approach improves sensitivity while preserving specificity.

🧬 Feature Categories

Features were selected based on clinical relevance and availability in early ICU care:

Category	Examples
Vital signs	Heart rate, SpO₂, respiratory rate, MAP
Temperature	Maximum temperature
Neurologic status	GCS category
Interventions	Vasopressors, CRRT, ventilation type
Demographics	Age, sex, race
Comorbidity	Elixhauser category
Missingness indicators	For key physiologic variables
⚙️ Repository Structure
```text
├── data/
│   ├── raw/                  # (Not shared) Source dataset
│   ├── processed/            # Cleaned and modeling-ready datasets
│
├── src/
│   ├── preprocessing/        # Data cleaning & feature selection
│   ├── descriptive/          # Table 1 generation
│   ├── modeling/
│   │   ├── standard/         # Two-stage ML pipeline
│   │   ├── balanced_ensemble/
│   │   │   ├── lr/           # Logistic ensemble
│   │   │   ├── rf/           # RF ensemble
│   │   │   └── xgb/          # XGB ensemble
│   ├── evaluation/           # Metrics, calibration, comparison
│   ├── interpretability/     # Odds ratios & permutation importance
│
├── results/                  # Model outputs, tables, figures
├── requirements.txt
└── README.md



📊 Evaluation Metrics

Models are assessed using:

Discrimination

AUROC

AUPRC

Threshold-based performance

Sensitivity

Specificity

Precision

F1-score

Calibration

Brier score

Calibration curves

Calibration slope & intercept

All evaluations were performed on patient-level held-out test data.

🔍 Interpretability

To ensure clinical transparency:

Logistic Regression

Odds ratios with 95% confidence intervals

Forest plots for visual interpretation

Ensemble Models

Permutation feature importance (ΔAUROC)

Landmark-specific importance patterns

▶️ How to Run
1️⃣ Clone repository
git clone https://github.com/yourusername/sepsis-landmark-prediction.git
cd sepsis-landmark-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run pipeline step-by-step

Preprocessing → Modeling → Evaluation → Interpretability

Example:

python src/modeling/standard/run_two_stage_models.py
python src/modeling/balanced_ensemble/lr/run_balanced_lr.py
python src/evaluation/calibration_balanced_rf_lm18.py

📦 Requirements

All required Python libraries are listed in requirements.txt

Main dependencies:

pandas

numpy

scikit-learn

xgboost

matplotlib

scipy

joblib

🔐 Data Availability

Due to data use agreements (e.g., MIMIC-IV), the raw dataset cannot be shared.
Scripts are designed to run on similarly structured ICU datasets.

📖 Citation

If you use this codebase in your research, please cite the associated manuscript (under preparation).

📜 License

This project is licensed under the MIT License — you are free to use, modify, and distribute with attribution.

🤝 Acknowledgment

Developed as part of academic research in clinical risk prediction and interpretable machine learning for critical care.
