from fastapi import APIRouter
from .schema import ChurnRequest
from .service import predict_churn

router = APIRouter(prefix="/churn", tags=["Churn Prediction"])


@router.post("/predict")
def churn_predict(request: ChurnRequest):
    return predict_churn(request.dict())
