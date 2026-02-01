import joblib
import pandas as pd
import os

# --------------------------------------------------
# Resolve paths
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models_fraud")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model_v1.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "fraud_threshold.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "fraud_feature_columns.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "fraud_amount_scaler.pkl")


# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)
feature_columns = joblib.load(FEATURE_PATH)
scaler = joblib.load(SCALER_PATH)


# --------------------------------------------------
# Risk Logic
# --------------------------------------------------
def get_risk_level(prob):
    if prob >= 0.95:
        return "CRITICAL"
    elif prob >= 0.85:
        return "HIGH"
    elif prob >= 0.60:
        return "MEDIUM"
    else:
        return "LOW"


# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict_fraud(data: dict):

    df = pd.DataFrame([data])

    # Scale Amount
    df["Amount_scaled"] = scaler.transform(df[["Amount"]])
    df = df.drop(columns=["Amount"])

    # Align features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    probability = float(model.predict_proba(df)[0][1])

    will_block = bool(probability >= float(threshold))
    risk_level = get_risk_level(probability)

    return {
        "fraud_probability": round(probability, 6),
        "threshold_used": float(threshold),
        "will_block_transaction": will_block,
        "risk_level": risk_level
    }
