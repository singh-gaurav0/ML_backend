import joblib
import numpy as np
import pandas as pd

from app.shared.feature_engineering import compute_house_age

MODEL_PATH = "app/models/house_price/house_model.pkl"
PREPROCESSOR_PATH = "app/models/house_price/preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

RMSE_LOG = 0.24   # from evaluation

def clean_feature_name(name: str) -> str:
    return (
        name.replace("num__", "")
            .replace("_", " ")
            .strip()
            .title()
    )
def predict_house_price(payload: dict):
    # 1️⃣ Feature engineering + column mapping
    df = compute_house_age(payload)

    # 2️⃣ Preprocess
    processed = preprocessor.transform(df)

    # 3️⃣ Predict (log space → real space)
    log_pred = model.predict(processed)[0]
    prediction = np.expm1(log_pred)

    # 4️⃣ Confidence interval
    confidence = 1.96 * RMSE_LOG
    lower = np.expm1(log_pred - confidence)
    upper = np.expm1(log_pred + confidence)

    # =========================
    # 🔥 Explainability section
    # =========================

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_

    # contribution in log space
    contributions = processed[0] * coefficients

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions
    }).sort_values(by="contribution", ascending=False)

    # approximate conversion to price impact
    total_log = explanation_df["contribution"].sum() + model.intercept_
    explanation_df["relative_impact"] = explanation_df["contribution"] / total_log
    explanation_df["approx_price_impact"] = explanation_df["relative_impact"] * prediction

    top_positive = [

        {
            "feature": clean_feature_name(row["feature"]),
            "approx_price_impact": round(row["approx_price_impact"], 2)
        }

        for _, row in (
            explanation_df[explanation_df["approx_price_impact"] > 0]
            .head(3)
            .iterrows()
        )
    ]

    top_negative = [
        {
            "feature": clean_feature_name(row["feature"]),
            "approx_price_impact": round(row["approx_price_impact"], 2)
        }
        for _, row in (
            explanation_df[explanation_df["approx_price_impact"] < 0]
            .tail(3)
            .iterrows()
        )
    ]

    # 5️⃣ Final response
    return {
        "predicted_price": round(prediction, 2),
        "confidence_range": {
            "lower": round(lower, 2),
            "upper": round(upper, 2)
        },
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative
    }
