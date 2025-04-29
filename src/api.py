import pandas as pd
from aiohttp.abc import HTTPException
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from src.utils import CompanyData, save_predictions_file
from src.preprocessing import preprocessing_company_data
from src.model import predict_insurance_labels
from src.config import settings
import asyncio


router = APIRouter()


@router.post("/predict_company", status_code=200)
async def predict_company(data: CompanyData):
    df_data = pd.DataFrame([data.dict()])
    data_preprocessing = await preprocessing_company_data(data=df_data)
    predict_taxonomy = await predict_insurance_labels(data_preprocessing)

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
        raise HTTPException(400, f"Could not read CSV: {e}")

    df_clean = await preprocessing_company_data(data=df)

    # Run predictions in batches:
    async def run_and_save():
        batch_size = settings.model_params.batch_size
        preds = []
        # Loop in batches:
        for i in range(0, len(df_clean), batch_size):
            batch = df_clean[i : i + batch_size]
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda b=batch: predict_insurance_labels(b)
            )
            preds.extend(await result)
        print(f"\nData predicted is: {preds}\n")
        df_clean['insurance_label'] = preds
        save_predictions_file(df_clean)

    background_tasks.add_task(run_and_save)

    return {
        "success": True,
        "message": "File received. Predictions will be saved in background.",
        "output_dir": settings.preprocess.output_dir,
    }



