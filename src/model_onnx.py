from transformers import AutoTokenizer
from src.utils import candidate_labels
from typing import List
from pathlib import Path
import onnxruntime as ort
import numpy as np
import asyncio


tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-1")
onnx_path = Path(__file__).parent.parent / "output.onnx"
ort_session = ort.InferenceSession(str(onnx_path))



def prepare_inputs(text: str):
    """
    Tokenizes text and transforms input data into an ONNX compatible format.
    """
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)

    return inputs



def onnx_inference(inputs: dict):
    """
    Runs the ONNX model and returns the raw results.
    """
    outputs = ort_session.run(None, inputs)

    return outputs



def postprocess_outputs(outputs, candidate_labels: List[str], top_k):
    """
    This function should map the outputs of the ONNX model to labels.
    Here's the important thing: the exact structure of the outputs depends on the conversion.
    For example, suppose outputs[0] contains logits for candidate labels.
    """
    logits = outputs[0][0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    label_score_pairs = list(zip(candidate_labels, probabilities))
    sorted_labels = sorted(label_score_pairs, key=lambda x: x[1], reverse=True)
    selected_labels = [label for label, score in sorted_labels[:top_k]]

    return selected_labels



async def extract_labels_onnx(text: str, threshold: float = 0.7, top_k: int = 3) -> List[str]:
    """
    Asynchronous function that prepares the input, runs inference on the ONNX model, and extracts relevant labels.
    """
    loop = asyncio.get_running_loop()
    inputs = prepare_inputs(text)
    outputs = await loop.run_in_executor(None, lambda: onnx_inference(inputs))
    selected_labels = postprocess_outputs(outputs, candidate_labels, top_k)


    return selected_labels



async def predict_insurance_labels(full_text: str, threshold: float = 0.7, top_k: int = 3) -> List[str]:
    return await extract_labels_onnx(full_text, threshold=threshold, top_k=top_k)


