import joblib
import shap
import pandas as pd
import os

# ---------------------------
# Resolve project root
# ---------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# app/models/churn_prediction/service.py
# → project root
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models_churn", "churn_prediction")

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model_time_v6.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "churn_feature_columns.pkl")

# ---------------------------
# Load model + feature schema
# ---------------------------
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

explainer = shap.TreeExplainer(model)

# ---------------------------
# Prediction Function
# ---------------------------
def predict_churn(data: dict):

    df = pd.DataFrame([data])

    # Align with training schema
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    probability = float(model.predict_proba(df)[0][1])

    # ---------------------------
    # Risk decision layer
    # ---------------------------
    if probability >= 0.80:
        risk_level = "CRITICAL"
        will_churn_soon = True
    elif probability >= 0.65:
        risk_level = "HIGH"
        will_churn_soon = True
    elif probability >= 0.50:
        risk_level = "MEDIUM"
        will_churn_soon = False
    else:
        risk_level = "LOW"
        will_churn_soon = False

    # ---------------------------
    # SHAP explanation
    # ---------------------------
    shap_values = explainer.shap_values(df)[0]

    shap_df = pd.DataFrame({
        "feature": feature_columns,
        "impact": shap_values
    })

    shap_df["abs_impact"] = shap_df["impact"].abs()
    shap_df = shap_df.sort_values("abs_impact", ascending=False)

    top_positive = (
        shap_df[shap_df["impact"] > 0]
        .head(3)[["feature", "impact"]]
        .to_dict(orient="records")
    )

    top_negative = (
        shap_df[shap_df["impact"] < 0]
        .head(3)[["feature", "impact"]]
        .to_dict(orient="records")
    )

    return {
        "churn_probability": round(probability, 4),
        "will_churn_soon": will_churn_soon,
        "risk_level": risk_level,
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative
    }
