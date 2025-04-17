import asyncio
from typing import List
from transformers import pipeline
from src.utils import candidate_labels


classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", device=0)


async def extract_labels(result, threshold: float = 0.7, top_k: int = 3) -> List[str]:
    selected = [label for label, score in zip(result['labels'], result['scores']) if score > threshold]
    if not selected:
        selected = [result['labels'][0]]
    return selected[:top_k]


async def predict_insurance_labels(full_text: str, threshold: float = 0.7, top_k: int = 3) -> List[str]:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: classifier(full_text, candidate_labels=candidate_labels, multi_label=True, backsize=10)
    )

    return await extract_labels(result, threshold=threshold, top_k=top_k)
