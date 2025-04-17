import pandas as pd
from fastapi import APIRouter
from src.utils import CompanyData
from src.preprocessing import preprocessing_company_data
from src.model import predict_insurance_labels


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
