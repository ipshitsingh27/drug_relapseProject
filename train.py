import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("patient_drug_relapse_dataset.csv")

# Drop ID column
df = df.drop(columns=["patient_id"])

# Define target
y = df["relapsed"]

# Define features (everything except target)
X = df.drop(columns=["relapsed"]).copy()

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build model
model = Sequential([
    LSTM(64, input_shape=(1, X_scaled.shape[2])),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save everything
model.save("drug_relapse_lstm_model.keras")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Model trained and saved successfully.")
