import pandas as pd

# API field → training column name mapping
COLUMN_MAPPING = {
    "number_of_bedrooms": "number of bedrooms",
    "number_of_bathrooms": "number of bathrooms",
    "living_area": "living area",
    "lot_area": "lot area",
    "number_of_floors": "number of floors",
    "waterfront_present": "waterfront present",
    "number_of_views": "number of views",
    "condition_of_the_house": "condition of the house",
    "grade_of_the_house": "grade of the house",
    "Area_of_the_house_excluding_basement": "Area of the house(excluding basement)",
    "Area_of_the_basement": "Area of the basement",
    "living_area_renov": "living_area_renov",
    "lot_area_renov": "lot_area_renov",
    "Number_of_schools_nearby": "Number of schools nearby",
    "Distance_from_the_airport": "Distance from the airport",
}

def compute_house_age(input_data: dict) -> pd.DataFrame:
    # --- extract year fields ---
    built_year = input_data.pop("built_year")
    renovation_year = input_data.pop("renovation_year", built_year)

    if renovation_year == 0 or renovation_year is None:
        renovation_year = built_year

    current_year = 2025
    house_age = current_year - built_year
    years_since_renovation = current_year - renovation_year

    # --- map API fields to training column names ---
    mapped_features = {}

    for api_key, value in input_data.items():
        if api_key not in COLUMN_MAPPING:
            raise ValueError(f"Unexpected field received: {api_key}")
        mapped_features[COLUMN_MAPPING[api_key]] = value

    # --- add engineered features (exact training names) ---
    mapped_features["house_age"] = house_age
    mapped_features["years_since_renovation"] = years_since_renovation

    return pd.DataFrame([mapped_features])
