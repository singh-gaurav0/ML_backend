from fastapi import FastAPI
from app.models.house_price.router import router as house_router
from app.models.salary_prediction.router import router as salary_router
from app.models.churn_prediction.router import router as churn_router
from app.models.fraud_detection.router import router as fraud_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="ML Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # local dev
        "https://ml-project-portfolio.vercel.app",  # later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(house_router)
app.include_router(salary_router)
app.include_router(churn_router)
app.include_router(fraud_router)
@app.get("/")
def health():
    return {"status": "ok"}
