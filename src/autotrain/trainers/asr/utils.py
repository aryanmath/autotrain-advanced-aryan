import numpy as np
import torch
from datasets import Audio
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollatorMixin

from autotrain import logger


class ASRDataCollator(DataCollatorMixin):
    """
    Data collator for ASR tasks that dynamically pads audio inputs and labels.
    Adapted from TextClassificationDataCollator and TokenClassificationDataCollator.
    """
    def __init__(self, feature_extractor, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Extract input_values (audio features) and labels (tokenized transcripts)
        input_values = [feature["input_values"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad audio inputs
        batch = self.feature_extractor.pad(
            {"input_values": input_values},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels with tokenizer, handling CTC-specific requirements
        with self.tokenizer.as_target_tokenizer():
            labels_batch = self.tokenizer.pad(
                {"input_ids": labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

        # Replace padding with -100 for CTC loss (ignore in loss computation)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


def preprocess_audio_dataset(dataset, feature_extractor, tokenizer, audio_column="audio", text_column="text"):
    """
    Preprocess audio dataset by extracting features and tokenizing transcripts.
    Inspired by image_classification and clm utils.
    """
    # Ensure audio column is in the correct format
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=feature_extractor.sampling_rate))

    def preprocess_function(examples):
        # Extract audio features
        audio_inputs = feature_extractor(
            [audio["array"] for audio in examples[audio_column]],
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="np",
            padding=True,
        ).input_values

        # Tokenize transcripts
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="np",
            ).input_ids

        return {
            "input_values": audio_inputs,
            "labels": labels,
        }

    logger.info("Preprocessing audio dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing ASR dataset",
    )
    return processed_dataset


def compute_metrics(eval_pred: EvalPrediction, tokenizer, metric):
    """
    Compute WER (Word Error Rate) for ASR evaluation.
    Adapted from seq2seq and token_classification utils.
    """
    predictions, labels = eval_pred
    # Predictions are logits, take argmax to get token IDs
    predictions = np.argmax(predictions, axis=-1)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
    wer = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"wer": wer}


def load_asr_components(model_name_or_path):
    """
    Load feature extractor, tokenizer, and model for ASR.
    Inspired by clm and text_classification utils.
    """
    logger.info(f"Loading ASR components from {model_name_or_path}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return feature_extractor, tokenizer


def save_to_hub(model, feature_extractor, tokenizer, username, token, project_name):
    """
    Save ASR model and components to Hugging Face Hub.
    Adapted from clm and dreambooth utils.
    """
    if username and token:
        repo_id = f"{username}/{project_name}"
        logger.info(f"Pushing to Hugging Face Hub: {repo_id}")
        model.push_to_hub(repo_id=repo_id, token=token)
        feature_extractor.push_to_hub(repo_id=repo_id, token=token)
        tokenizer.push_to_hub(repo_id=repo_id, token=token)
    else:
        logger.info("Hub credentials not provided, skipping push to Hub.")


def post_process_predictions(predictions, tokenizer):
    """
    Post-process ASR predictions for evaluation or inference.
    Inspired by seq2seq utils.
    """
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    return [pred.strip() for pred in decoded_preds]