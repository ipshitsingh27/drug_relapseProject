import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load scaler and encoders
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load trained model
model = load_model("drug_relapse_lstm_model.keras", compile=False)

st.title("💊 Drug Relapse Prediction App")

# User Inputs
age = st.number_input("Age", 10, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
primary_drug = st.selectbox("Primary Drug", ["Alcohol", "Cocaine", "Heroin", "Cannabis", "Meth"])
usage_frequency_per_week = st.number_input("Usage Frequency Per Week", 0, 50, 3)
usage_duration_months = st.number_input("Usage Duration (Months)", 0, 240, 12)
mental_health_score = st.slider("Mental Health Score (0-100)", 0, 100, 50)
social_support_score = st.slider("Social Support Score (0-100)", 0, 100, 50)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student"])
previous_relapses = st.number_input("Previous Relapses", 0, 20, 0)
treatment_attempts = st.number_input("Treatment Attempts", 0, 20, 0)
days_since_last_use = st.number_input("Days Since Last Use", 0, 3650, 30)

# Create DataFrame
input_data = {
    "age": [age],
    "gender": [gender],
    "primary_drug": [primary_drug],
    "usage_frequency_per_week": [usage_frequency_per_week],
    "usage_duration_months": [usage_duration_months],
    "mental_health_score": [mental_health_score],
    "social_support_score": [social_support_score],
    "employment_status": [employment_status],
    "previous_relapses": [previous_relapses],
    "treatment_attempts": [treatment_attempts],
    "days_since_last_use": [days_since_last_use]
}

input_df = pd.DataFrame(input_data)

# Encode categorical columns
for col in input_df.select_dtypes(include="object").columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Ensure column order matches training
input_df = input_df[scaler.feature_names_in_]

# Scale
input_scaled = scaler.transform(input_df)

# Reshape for LSTM
input_scaled = input_scaled.reshape(1, 1, input_scaled.shape[1])

# Predict
prediction = model.predict(input_scaled)
risk = prediction[0][0]

st.subheader("Prediction Result")

if risk > 0.5:
    st.error(f"⚠️ High Risk of Relapse ({risk:.2f})")
else:
    st.success(f"✅ Low Risk of Relapse ({risk:.2f})")
