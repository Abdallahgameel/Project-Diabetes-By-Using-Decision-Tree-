from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "diabetes_model.joblib"

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model, features, threshold = artifact["model"], artifact["features"], artifact["threshold"]

st.title("🩺 Diabetes Risk Prediction")
st.warning(
    "⚠️ This is an experimental tool built on BRFSS 2015 survey data. "
    "It is NOT a medical diagnosis. Please consult a healthcare professional."
)

AGE_LABELS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
              "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
GENHLTH = {"1 - Excellent": 1, "2 - Very good": 2, "3 - Good": 3, "4 - Fair": 4, "5 - Poor": 5}
EDUCATION = {
    "1 - Never attended / kindergarten only": 1,
    "2 - Elementary": 2,
    "3 - Some high school": 3,
    "4 - High school graduate / GED": 4,
    "5 - Some college (1-3 years)": 5,
    "6 - College graduate (4+ years)": 6,
}


def yes_no(label):
    return 1 if st.selectbox(label, ["No", "Yes"], key=label) == "Yes" else 0


col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Health")
    bmi = st.number_input("BMI", 12, 98, 27)
    high_bp = yes_no("High blood pressure?")
    high_chol = yes_no("High cholesterol?")
    chol_check = yes_no("Cholesterol check in the last 5 years?")
    stroke = yes_no("Ever had a stroke?")
    heart = yes_no("Coronary heart disease / heart attack?")
    diff_walk = yes_no("Serious difficulty walking / climbing stairs?")

with col2:
    st.subheader("General status & lifestyle")
    gen_hlth = GENHLTH[st.selectbox("General health", list(GENHLTH))]
    phys_hlth = st.slider("Days of poor physical health (last 30 days)", 0, 30, 0)
    ment_hlth = st.slider("Days of poor mental health (last 30 days)", 0, 30, 0)
    smoker = yes_no("Smoked at least 100 cigarettes in your life?")
    phys_act = yes_no("Physical activity in the last 30 days?")
    fruits = yes_no("Eat fruit daily?")
    veggies = yes_no("Eat vegetables daily?")
    alcohol = yes_no("Heavy alcohol consumption?")

with col3:
    st.subheader("Demographics & access to care")
    sex = 1 if st.selectbox("Sex", ["Female", "Male"]) == "Male" else 0
    age = AGE_LABELS.index(st.selectbox("Age group", AGE_LABELS)) + 1
    education = EDUCATION[st.selectbox("Education", list(EDUCATION))]
    income = st.slider("Income level (1 = lowest, 8 = highest)", 1, 8, 5)
    any_health = yes_no("Have any healthcare coverage?")
    no_doc = yes_no("Could not see a doctor because of cost (last 12 months)?")

row = {
    "HighBP": high_bp, "HighChol": high_chol, "CholCheck": chol_check, "BMI": bmi,
    "Smoker": smoker, "Stroke": stroke, "HeartDiseaseorAttack": heart,
    "PhysActivity": phys_act, "Fruits": fruits, "Veggies": veggies,
    "HvyAlcoholConsump": alcohol, "AnyHealthcare": any_health, "NoDocbcCost": no_doc,
    "GenHlth": gen_hlth, "MentHlth": ment_hlth, "PhysHlth": phys_hlth,
    "DiffWalk": diff_walk, "Sex": sex, "Age": age, "Education": education, "Income": income,
}

if st.button("Predict", type="primary"):
    user_df = pd.DataFrame([row])[features]  # same column order as training
    proba = model.predict_proba(user_df)[0, 1]

    st.divider()
    st.metric("Model score (probability of class 1)", f"{proba:.1%}")
    if proba >= threshold:
        st.error("Flagged: this profile may need an additional health assessment.")
    else:
        st.success("Not flagged by the model. This does not rule out diabetes.")
    st.caption(f"Decision threshold used: {threshold:.2f}. The class includes prediabetes and diabetes.")
