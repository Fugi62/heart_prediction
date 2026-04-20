import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Load model files (SAFE METHOD)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "knn_heart.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
expected_columns = joblib.load(os.path.join(BASE_DIR, "heart_columns.pkl"))

# -------------------------------
# UI Configuration
# -------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.markdown("Provide the following details to check your heart disease risk:")

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("🧍 Personal Details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])

st.subheader("🫀 Medical Details")

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):

    # Basic numeric input
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
    }

    # Categorical inputs
    categorical_inputs = {
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingECG': resting_ecg,
        'ExerciseAngina': exercise_angina,
        'ST_Slope': st_slope
    }

    # One-hot encoding
    for feature, value in categorical_inputs.items():
        col_name = f"{feature}_{value}"
        raw_input[col_name] = 1

    # Create DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Probability (if supported)
    try:
        prob = model.predict_proba(scaled_input)[0][1]
    except:
        prob = None

    # -------------------------------
    # Output
    # -------------------------------
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({prob:.2f})" if prob else "⚠️ High Risk of Heart Disease")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({prob:.2f})" if prob else "✅ Low Risk of Heart Disease")

    # Disclaimer
    st.warning("⚠️ This prediction is for educational purposes only and not a medical diagnosis. Please consult a doctor.")
