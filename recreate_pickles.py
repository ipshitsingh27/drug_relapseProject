import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("patient_drug_relapse_dataset.csv")

# Automatically detect categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Automatically detect numeric columns
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ scaler.pkl and label_encoders.pkl recreated successfully")
