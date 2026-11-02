# Diabetes Prediction Using Machine Learning

## Project Overview

This project focuses on developing a Machine Learning model to predict diabetes status based on health-related indicators from the **2015 Behavioral Risk Factor Surveillance System (BRFSS)** dataset.

The project uses a Decision Tree Classifier to identify patterns in health and lifestyle indicators and classify individuals into three categories:

- **0 – No Diabetes**
- **1 – Prediabetes**
- **2 – Diabetes**

In addition to the original multiclass classification task, the project explores binary diabetes detection by applying a custom probability threshold to identify individuals belonging to the Diabetes class.

## Objectives

- Explore and understand health-related data.
- Perform Exploratory Data Analysis (EDA).
- Identify patterns and relationships between health indicators and diabetes status.
- Build and evaluate Machine Learning classification models.
- Handle class imbalance and evaluate model performance using appropriate metrics.
- Analyze model errors and performance across different age and BMI groups.
- Deploy the trained model through an interactive Streamlit web application.

## Dataset

**Source:** Behavioral Risk Factor Surveillance System (BRFSS) 2015.

The dataset contains health-related indicators, including:

- BMI
- High Blood Pressure
- High Cholesterol
- Physical Activity
- Smoking
- Age
- General Health
- Heart Disease or Attack

**Target Variable:** `Diabetes_012`

| Value | Description |
|---|---|
| 0 | No Diabetes |
| 1 | Prediabetes |
| 2 | Diabetes |

## Project Workflow

1. **Data Understanding:** Explore the dataset structure, features, and target distribution.
2. **Data Cleaning:** Handle duplicate records and inspect data quality.
3. **Exploratory Data Analysis:** Analyze distributions, relationships, and potential risk indicators.
4. **Preprocessing:** Prepare features for Machine Learning.
5. **Model Development:** Train and evaluate Decision Tree models.
6. **Hyperparameter Tuning:** Use GridSearchCV and Stratified Cross-Validation.
7. **Model Evaluation:** Evaluate performance using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
8. **Threshold Optimization:** Select a probability threshold to prioritize Recall when detecting the Diabetes class.
9. **Error Analysis:** Investigate model performance across age and BMI groups.
10. **Deployment:** Build an interactive web application using Streamlit.

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Web Application

The trained model is saved using Joblib and integrated into a Streamlit application, allowing users to enter health-related information and receive a model prediction.

## Project Structure

```text
Diabetes-Prediction/
│
├── files/
│   └── app.py
│
├── models/
│   └── diabetes_model.joblib
│
├── notebooks/
│   └── diabetes_prediction.ipynb
│
├── README.md
└── requirements.txt
```

## How to Run the Project

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Diabetes-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run files/app.py
```

## Important Limitations

- This project is an educational Machine Learning prototype, not a medical diagnostic tool.
- The dataset is based on the 2015 US BRFSS survey and may not represent other populations or current health conditions.
- Model predictions should not replace professional medical evaluation.
- Further validation using recent and independent datasets is required before considering real-world clinical use.

## Author

Developed as a Machine Learning project to practice data analysis, classification, model evaluation, and deployment using Python.