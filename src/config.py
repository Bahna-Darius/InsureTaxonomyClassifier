from pydantic import BaseModel
from functools import lru_cache
import yaml


class PreprocessParams(BaseModel):
    input_dir: str
    output_dir: str
    data_file: str


class ModelParams(BaseModel):
    batch_size: int
    model_name: str
    model_task: str
    model_device: int
    predict_threshold: float
    top_predict: int


class Settings(BaseModel):
    preprocess: PreprocessParams
    model_params: ModelParams


@lru_cache()
def load_settings(path: str = "config.yaml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Settings(**data)


settings = load_settings()
