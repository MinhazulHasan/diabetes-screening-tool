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

# Diabetes Prediction

A simple Gradio app that predicts diabetes risk from 8 medical measurements, trained on the Pima Indians Diabetes dataset.

## Dataset

Pima Indians Diabetes Database (768 patients, 8 features, target = Outcome).
Loaded via `kagglehub` from `uciml/pima-indians-diabetes-database`.

## Model

Logistic Regression with `class_weight='balanced'`, tuned via 5-fold StratifiedKFold and GridSearchCV.

Test set performance: Accuracy ~77%, Recall ~80%, ROC-AUC ~0.83.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

## Files

- `ML_Final_Exam_Pima_Diabetes.ipynb` — full notebook
- `app.py` — Gradio web app
- `diabetes_model.pkl` — trained pipeline
- `metadata.pkl` — UI defaults and metrics
- `diabetes.csv` — dataset
- `requirements.txt` — dependencies
