import pandas as pd
from aiohttp.abc import HTTPException
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from src.utils import CompanyData, save_predictions_file
from src.preprocessing import preprocessing_company_data
from src.model import predict_batch
from src.config import settings
import asyncio


router = APIRouter()


@router.post("/predict_company", status_code=200)
async def predict_company(data: CompanyData):
    df_data = pd.DataFrame([data.dict()])
    data_preprocessing = await preprocessing_company_data(data=df_data)
    texts = data_preprocessing["full_text"].tolist()
    predict_taxonomy = await predict_batch(
        texts,
        threshold=settings.model_params.predict_threshold,
        top_k=settings.model_params.top_predict
    )

    return {
        "insurance_label": predict_taxonomy,
        "result": {
            "success": True,
            "code": 200,
            "message": "Predict model succesul!"
        },
    }

@router.post("/predict_file", status_code=200)
async def predict_file(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
):

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    df_clean = await preprocessing_company_data(df)

    def run_and_save(df_input: pd.DataFrame):
        batch_size = settings.model_params.batch_size
        all_preds = []
        texts = df_input["full_text"].tolist()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_labels = asyncio.run(
                predict_batch(
                    batch_texts,
                    threshold=settings.model_params.predict_threshold,
                    top_k=settings.model_params.top_predict
                )
            )
            all_preds.extend(batch_labels)

        df_input["insurance_label"] = all_preds
        save_predictions_file(df_input)

    background_tasks.add_task(run_and_save, df_clean)

    return {
        "success": True,
        "message": "File received. Predictions will be saved in background.",
        "output_dir": settings.preprocess.output_dir,
    }

