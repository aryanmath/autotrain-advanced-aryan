# import numpy as np
# from datasets import load_metric


# def compute_metrics(pred):
#     """
#     Compute Word Error Rate (WER) for ASR predictions.

#     Args:
#         pred: Predictions from the model.

#     Returns:
#         dict: A dictionary containing the WER score.
#     """
#     metric = load_metric("wer")
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)

#     pred_str = pred.processor.batch_decode(pred_ids)
#     label_str = pred.label_ids

#     wer = metric.compute(predictions=pred_str, references=label_str)
#     return {"wer": wer}




import evaluate
import numpy as np

def compute_metrics(eval_pred):
    """
    Compute metrics for ASR evaluation.
    Args:
        eval_pred: Tuple containing predictions and labels.
    Returns:
        Dictionary with WER (Word Error Rate).
    """
    wer_metric = evaluate.load("wer")
    predictions, labels = eval_pred

    # Replace -100 in labels (used for padding) with pad_token_id
    predictions = np.where(predictions != -100, predictions, wer_metric.pad_token_id)
    labels = np.where(labels != -100, labels, wer_metric.pad_token_id)

    # Decode predictions and labels to text
    decoded_preds = [wer_metric.tokenizer.decode(pred) for pred in predictions]
    decoded_labels = [wer_metric.tokenizer.decode(label) for label in labels]

    # Compute WER
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"wer": wer}