import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load scaler and label encoders
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load your trained model
model = load_model("drug_relapse_lstm_model.keras")

st.title("💊 Drug Relapse Prediction App")
st.write("Fill in the details below to predict the risk of drug relapse.")

# User input
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
years_of_use = st.number_input("Years of Drug Use", min_value=0, max_value=50, value=1)
support_system = st.selectbox("Support System", ["Weak", "Moderate", "Strong"])

# Prepare input DataFrame
input_data = {
    "age": [age],
    "gender": [gender],
    "years_of_use": [years_of_use],
    "support_system": [support_system]
}

input_df = pd.DataFrame(input_data)

# Only encode columns present in input
for col in input_df.select_dtypes(include="object").columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Scale input
feature_columns = ["age", "years_of_use", "gender", "support_system"]
input_scaled = scaler.transform(input_df[feature_columns])

# Predict
prediction = model.predict(input_scaled)
risk = prediction[0][0]

st.subheader("Prediction Result")
if risk > 0.5:
    st.error(f"⚠️ High risk of drug relapse ({risk:.2f})")
else:
    st.success(f"✅ Low risk of drug relapse ({risk:.2f})")
