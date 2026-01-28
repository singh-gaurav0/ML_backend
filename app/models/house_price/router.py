from fastapi import APIRouter
from .schema import HouseInput, HousePredictionResponse
from .service import predict_house_price

router = APIRouter(prefix="/house-price", tags=["House Price"])

@router.post("/predict", response_model=HousePredictionResponse)
def predict(data: HouseInput):
    return predict_house_price(data.dict())
