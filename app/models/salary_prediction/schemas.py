from pydantic import BaseModel
from typing import List, Dict


class SalaryRequest(BaseModel):
    Education: str
    Experience: int
    Location: str
    Job_Title: str
    Age: int


class FactorImpact(BaseModel):
    feature: str
    approx_salary_impact: float


class ConfidenceRange(BaseModel):
    lower: float
    upper: float


class SalaryResponse(BaseModel):
    predicted_salary: float
    confidence_range: ConfidenceRange
    model_metrics: Dict[str, float]
    top_positive_factors: List[FactorImpact]
    top_negative_factors: List[FactorImpact]
