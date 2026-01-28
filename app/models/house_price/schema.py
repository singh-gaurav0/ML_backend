from pydantic import BaseModel
from typing import Optional

class HouseInput(BaseModel):
    number_of_bedrooms: int
    number_of_bathrooms: float
    living_area: float
    lot_area: float
    number_of_floors: float
    condition_of_the_house: int
    grade_of_the_house: int
    Area_of_the_house_excluding_basement: float
    Area_of_the_basement: float
    living_area_renov: float
    lot_area_renov: float
    Number_of_schools_nearby: int
    Distance_from_the_airport: float
    built_year: int

    renovation_year: Optional[int] = None
    waterfront_present: Optional[int] = 0
    number_of_views: Optional[int] = 0


class HousePredictionResponse(BaseModel):
    predicted_price: float
    confidence_range: dict
    top_positive_factors: list
    top_negative_factors: list
