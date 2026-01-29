import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter

from app.models.salary_prediction.schemas import (
    SalaryRequest,
    SalaryResponse
)

router = APIRouter(prefix="/salary", tags=["Salary Prediction"])

model = joblib.load("app/models/salary_prediction/salary_pipeline.pkl")
metadata = joblib.load("app/models/salary_prediction/salary_metadata.pkl")
residual_std = joblib.load("app/models/salary_prediction/salary_residual_std.pkl")

feature_names = metadata["feature_names"]
coefficients = model.named_steps["regressor"].coef_
intercept = model.named_steps["regressor"].intercept_


def confidence_interval(prediction: float, std: float, z: float = 1.96):
    return {
        "lower": prediction - z * std,
        "upper": prediction + z * std
    }


@router.post("/predict", response_model=SalaryResponse)
def predict_salary(payload: SalaryRequest):

    input_df = pd.DataFrame([payload.dict()])

    predicted_salary = float(model.predict(input_df)[0])

    transformed = model.named_steps["preprocessing"].transform(input_df)
    contributions = transformed.flatten() * coefficients

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "impact": contributions
    })

    explanation_df["abs_impact"] = explanation_df["impact"].abs()
    explanation_df = explanation_df.sort_values("abs_impact", ascending=False)

    top_positive = (
        explanation_df[explanation_df["impact"] > 0]
        .head(3)[["feature", "impact"]]
    )

    top_negative = (
        explanation_df[explanation_df["impact"] < 0]
        .head(3)[["feature", "impact"]]
    )

    return {
        "predicted_salary": round(predicted_salary, 2),
        "confidence_range": confidence_interval(predicted_salary, residual_std),
        "model_metrics": {
            "mae": 8125.73,
            "rmse": 10300.71,
            "r2_score": 0.87
        },
        "top_positive_factors": [
            {
                "feature": row.feature,
                "approx_salary_impact": round(row.impact, 2)
            }
            for row in top_positive.itertuples()
        ],
        "top_negative_factors": [
            {
                "feature": row.feature,
                "approx_salary_impact": round(row.impact, 2)
            }
            for row in top_negative.itertuples()
        ]
    }
