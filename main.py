from fastapi import FastAPI, APIRouter
from src.api import router as prediction_router

app = FastAPI(
    title="Insurance Company Classifier API"
)

api_router = APIRouter()

api_router.include_router(
    prediction_router,
    prefix="/prediction",
    tags=["predict company"]
)

app.include_router(api_router)
