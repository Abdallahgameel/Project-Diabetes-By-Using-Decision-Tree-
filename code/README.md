# 🩺 Diabetes Risk Prediction (Decision Tree)

## Project Overview
End-to-end ML project: EDA → preprocessing → Decision Tree tuning → evaluation → Streamlit web app.

## Problem Statement
Using self-reported health, lifestyle and demographic indicators (BRFSS 2015), flag people who may need an
additional diabetes screening. Target: `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes).
Main metric: **Recall**, while monitoring **Precision**.

## Dataset Source
CDC BRFSS 2015 — Diabetes Health Indicators (original imbalanced version, ~253k rows, 21 features).
Download link + instructions: `data/README.md`.

## EDA Highlights
- _Add 3–5 insights with real numbers from your notebook._

## Model Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (Dummy) | | | | | |
| Decision Tree (tuned) | | | | | |

## Live App
🔗 _Paste your Streamlit Community Cloud link here_

![App demo](images/app_demo.png)

## Installation & Usage
```bash
pip install -r requirements.txt
# 1) put the CSV in data/  2) run notebooks/diabetes_analysis.ipynb (creates models/diabetes_model.joblib)
streamlit run app/app.py
```

## Limitations
BRFSS 2015 data, self-reported answers, correlation ≠ causation, experimental tool — not a medical diagnosis.
