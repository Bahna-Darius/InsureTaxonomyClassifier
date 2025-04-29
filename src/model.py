import asyncio
from typing import List
from transformers import pipeline
from src.utils import candidate_labels
from src.config import settings


classifier = pipeline(
    settings.model_params.model_task,
    model=settings.model_params.model_name,
    device=settings.model_params.model_device,
    batch_size=settings.model_params.batch_size
)


def _run_classification(texts: List[str], threshold: float, top_k: int):
    results = classifier(texts, candidate_labels=candidate_labels, multi_label=True)
    all_labels = []
    for r in results:
        labels = [lab for lab, sc in zip(r["labels"], r["scores"]) if sc > threshold]
        if not labels:
            labels = [r["labels"][0]]
        all_labels.append(labels[:top_k])
    return all_labels


async def predict_batch(
    texts: List[str],
    threshold: float = settings.model_params.predict_threshold,
    top_k: int  = settings.model_params.top_predict
) -> List[List[str]]:

    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        None,
        lambda: _run_classification(texts, threshold, top_k)
    )
