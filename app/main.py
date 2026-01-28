from fastapi import FastAPI
from app.models.house_price.router import router as house_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="ML Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # local dev
        "https://your-frontend-domain.vercel.app",  # later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(house_router)

@app.get("/")
def health():
    return {"status": "ok"}
