---
title: Diabetes Risk Screening Tool
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Diabetes Risk Screening Tool

A clinical decision-support tool that estimates diabetes risk from basic medical measurements. Designed for community health workers in resource-limited settings.

## Real-World Context

Bangladesh has more than 13 million adults with diabetes, with prevalence rising in rural areas where diagnostic labs are scarce. This tool enables frontline health workers (such as ASHA workers) to triage patients using only basic measurements (Glucose, BMI, Blood Pressure, Age), referring high-risk individuals for confirmatory lab testing.

## Project Overview

End-to-end ML pipeline covering:

1. Data Loading - Pima Indians Diabetes Database (NIH research, 768 patients)
2. Preprocessing - 6 distinct steps including hidden-zero detection (5 columns have biologically impossible 0 values that are actually missing data)
3. Pipeline - `ColumnTransformer` + `KNNImputer` for clinical-context-aware missing value handling
4. Model Selection - Compared LogReg, RandomForest and XGBoost with class imbalance handling
5. Cross-Validation - 5-fold StratifiedKFold with 5 metrics
6. Hyperparameter Tuning - GridSearchCV (20 combinations x 5 folds = 100 fits)
7. Evaluation - Comprehensive metrics including ROC, PR curves, clinical interpretation
8. Deployment - Gradio interface on Hugging Face Spaces

## Model Performance (Held-out Test Set, n=154)

| Metric | Value | Clinical Meaning |
|---|---|---|
| Accuracy | 0.7662 | Overall correctness |
| Precision (PPV) | 0.6324 | Of those flagged, % truly diabetic |
| Recall (Sensitivity) | 0.7963 | % of actual diabetics correctly caught |
| F1-Score | 0.7049 | Harmonic mean of precision and recall |
| ROC-AUC | 0.8348 | Discrimination across all thresholds |
| Average Precision | 0.7280 | Area under PR curve |

The key clinical achievement is 80% sensitivity. The model correctly identifies 4 out of 5 actual diabetics, making it suitable as a screening tool where the cost of missing a diabetic far exceeds the cost of a false positive (which only triggers a confirmatory test).

## Final Model Specifications

- Algorithm: Logistic Regression
- Class imbalance handling: `class_weight='balanced'`
- Regularization: L2 penalty, C = 1.0
- Solver: liblinear
- Selected via: GridSearchCV with F1 scoring (5-fold StratifiedKFold)

## Key Technical Highlights

### The "Hidden Zeros" Problem

The Pima dataset contains a known data quality trap: 5 columns have `0` values that are biologically impossible (a patient cannot have BloodPressure = 0 and still be alive). These are silent missing values.

| Column | Hidden zeros | % missing |
|---|---|---|
| Insulin | 374 | 48.7% |
| SkinThickness | 227 | 29.6% |
| BloodPressure | 35 | 4.6% |
| BMI | 11 | 1.4% |
| Glucose | 5 | 0.7% |

We convert these to NaN and impute using `KNNImputer(k=5)`, which uses similar patients to estimate missing values, more accurate than column-wide median for clinical data.

### Feature Engineering

Created 5 clinically meaningful features:

- `BMI_Category` - WHO standard categories
- `Age_Group` - diabetes risk rises with age
- `Glucose_Range` - ADA criteria (Normal / Prediabetic / Diabetic)
- `HighRisk_Score` - composite of Age x BMI x Pedigree
- `Insulin_BMI_Ratio` - insulin sensitivity proxy

## Repository Structure

```
.
├── app.py                              # Gradio web interface
├── requirements.txt                    # Python dependencies
├── diabetes_model.pkl                  # Trained sklearn pipeline
├── metadata.pkl                        # UI defaults and metric values
├── diabetes.csv                        # Pima Indians Diabetes dataset
├── ML_Final_Exam_Pima_Diabetes.ipynb   # Full notebook (all 11 tasks)
└── README.md                           # This file
```

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser.

## Dataset

Source: Pima Indians Diabetes Database

- Origin: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA
- Patients: 768 women of Pima Indian heritage, aged 21+
- Features: 8 medical measurements
- Target: Outcome (Diabetic Yes/No)
- Class imbalance: 1.87:1 (65.1% non-diabetic, 34.9% diabetic)

## Disclaimer

This is an educational machine learning project built for an academic course. Not for clinical use without proper medical validation, regulatory approval, and clinician oversight. Diagnosis of diabetes requires confirmatory laboratory testing.

## References

- Smith et al. (1988). "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus."
- WHO Diabetes Country Profile - Bangladesh (2023)
- IDF Diabetes Atlas, 10th Edition

---

*ML Final Exam Project - built with scikit-learn, KNN imputation, and Gradio*
