from fastapi import FastAPI, APIRouter
from src.api import router as prediction_router
from src.config import settings

app = FastAPI(
    title="Insurance Company Classifier API",
    description=f"Loaded with model={settings.model_params.model_name}, batch_size={settings.model_params.batch_size}"
)

api_router = APIRouter()

api_router.include_router(
    prediction_router,
    prefix="/prediction",
    tags=["predict company"]
)

app.include_router(api_router)
