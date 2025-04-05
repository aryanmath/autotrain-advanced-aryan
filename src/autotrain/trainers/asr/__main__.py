#!/usr/bin/env python

import argparse
import json
import os

import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCTC,
    Trainer,
    TrainingArguments,
)

from autotrain import logger
from autotrain.dataset import AutoTrainASRDataset
from autotrain.trainers.asr.params import ASRParams
from autotrain.trainers.asr.utils import (
    ASRDataCollator,
    preprocess_audio_dataset,
    compute_metrics,
    load_asr_components,
    save_to_hub,
)
from autotrain.trainers.common import monitor, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an ASR model with AutoTrain")
    parser.add_argument(
        "--training_config",
        type=str,
        required=True,
        help="Path to the training configuration JSON file",
    )
    return parser.parse_args()


@monitor
def train(config):
    # Load parameters from config
    params = ASRParams(**config)
    set_seed(params.seed)

    # Load dataset
    logger.info("Loading dataset...")
    if params.data_path.startswith("huggingface_hub:"):
        dataset_path = params.data_path.replace("huggingface_hub:", "")
        dataset = load_dataset(dataset_path, split=params.train_split)
        if params.valid_split:
            eval_dataset = load_dataset(dataset_path, split=params.valid_split)
        else:
            eval_dataset = None
    else:
        dataset_args = {
            "train_data": params.data_path,
            "token": params.token,
            "project_name": params.project_name,
            "username": params.username,
            "valid_data": None,
            "percent_valid": None,
            "local": True,
        }
        dset = AutoTrainASRDataset(**dataset_args)
        data_path = dset.prepare()
        dataset = load_dataset("audiofolder", data_dir=data_path, split="train")
        eval_dataset = None  # Validation split handled elsewhere if needed

    # Load feature extractor and tokenizer
    feature_extractor, tokenizer = load_asr_components(params.model)

    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    dataset = preprocess_audio_dataset(
        dataset,
        feature_extractor,
        tokenizer,
        audio_column=params.audio_column,
        text_column=params.text_column,
    )
    if eval_dataset:
        eval_dataset = preprocess_audio_dataset(
            eval_dataset,
            feature_extractor,
            tokenizer,
            audio_column=params.audio_column,
            text_column=params.text_column,
        )

    # Load model
    logger.info(f"Loading model: {params.model}")
    model = AutoModelForCTC.from_pretrained(
        params.model,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=params.project_name,
        num_train_epochs=params.epochs,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation,
        learning_rate=params.lr,
        warmup_steps=params.warmup_steps,
        max_steps=params.max_steps if params.max_steps > 0 else -1,
        eval_strategy=params.eval_strategy,
        save_strategy=params.eval_strategy,
        save_steps=params.save_steps,
        eval_steps=params.eval_steps,
        logging_steps=params.logging_steps,
        load_best_model_at_end=params.load_best_model_at_end,
        metric_for_best_model=params.metric_for_best_model,
        greater_is_better=params.greater_is_better,
        fp16=params.fp16 if torch.cuda.is_available() else False,
        gradient_checkpointing=params.gradient_checkpointing,
        save_total_limit=params.save_total_limit,
        push_to_hub=False,  # Handled separately via save_to_hub
        report_to=[params.log],
        disable_tqdm=False,
    )

    # Load WER metric
    wer_metric = evaluate.load("wer")

    # Initialize data collator
    data_collator = ASRDataCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,  # Feature extractor acts as tokenizer for inputs
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, wer_metric),
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    logger.info("Saving model...")
    trainer.save_model()
    feature_extractor.save_pretrained(params.project_name)
    tokenizer.save_pretrained(params.project_name)

    # Push to hub if configured
    if params.push_to_hub and params.username and params.token:
        logger.info("Pushing to Hugging Face Hub...")
        save_to_hub(
            model=trainer.model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            hub_username=params.username,
            hub_token=params.token,
            project_name=params.project_name,
        )

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.training_config):
        raise ValueError(f"Training config file not found: {args.training_config}")
    with open(args.training_config, "r") as f:
        config = json.load(f)
    train(config)