from fastapi import FastAPI
from app.models.house_price.router import router as house_router

app = FastAPI(title="ML Inference API")

app.include_router(house_router)

@app.get("/")
def health():
    return {"status": "ok"}
