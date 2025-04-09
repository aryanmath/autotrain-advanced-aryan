import numpy as np
from datasets import load_metric


def compute_metrics(pred):
    """
    Compute Word Error Rate (WER) for ASR predictions.

    Args:
        pred: Predictions from the model.

    Returns:
        dict: A dictionary containing the WER score.
    """
    metric = load_metric("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred_str = pred.processor.batch_decode(pred_ids)
    label_str = pred.label_ids

    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}