import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# === Load Dataset ===
df = pd.read_csv("UNSW_NB15_training-set.csv", low_memory=False)

# === Preprocessing ===
df = df.drop(columns=['id', 'attack_cat'])

# Encode categorical features
cat_cols = df.select_dtypes(include=['object']).columns
encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
for col in cat_cols:
    df[col] = encoders[col].transform(df[col])

# Features & Labels
X = df.drop(columns=['label'])
y = df['label']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# === Save model & scaler ===
joblib.dump(model, "rf_anomaly_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "label_encoders.pkl")
