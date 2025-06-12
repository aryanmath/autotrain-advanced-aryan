import os
import json
import logging
from typing import Dict, Any
import torch
from pathlib import Path

from autotrain import logger
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
from autotrain.trainers.automatic_speech_recognition.dataset import AutomaticSpeechRecognitionDataset
from autotrain.trainers.automatic_speech_recognition.utils import compute_metrics
from autotrain.trainers.common import monitor
from autotrain.logger import get_training_logger

class LifeAppASRTrainer:
    """
    Trainer for LiFE App ASR models.
    Reuses existing ASR code without modifying it.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.params = AutomaticSpeechRecognitionParams(**config)
        self.training_logger = get_training_logger(self.params.output_dir)
        
    def train(self):
        """Train the ASR model."""
        try:
            self.training_logger.info("Initializing LiFE App ASR training...")
            
            # Load dataset
            dataset = self._load_dataset()
            
            # Load model and processor
            model, processor = self._load_model_and_processor()
            
            # Create training dataset
            train_dataset = AutomaticSpeechRecognitionDataset(
                data=dataset,
                processor=processor,
                config=self.params,
                audio_column=self.params.audio_column,
                text_column=self.params.text_column,
                max_duration=self.params.max_duration,
                sampling_rate=self.params.sampling_rate,
            )
            
            # Create validation dataset if available
            valid_dataset = None
            if self.params.valid_split:
                valid_dataset = self._load_dataset(is_validation=True)
                valid_dataset = AutomaticSpeechRecognitionDataset(
                    data=valid_dataset,
                    processor=processor,
                    config=self.params,
                    audio_column=self.params.audio_column,
                    text_column=self.params.text_column,
                    max_duration=self.params.max_duration,
                    sampling_rate=self.params.sampling_rate,
                )
            
            # Train model
            self._train_model(model, processor, train_dataset, valid_dataset)
            
        except Exception as e:
            self.training_logger.error(f"Error during training: {str(e)}")
            raise
            
    def _load_dataset(self, is_validation: bool = False):
        """Load dataset from LiFE App JSON file."""
        try:
            self.training_logger.info(f"Loading dataset from LiFE App: {self.params.life_app_dataset_name}")
            dataset_path = os.path.join("src/autotrain/app/static", self.params.life_app_dataset_name)
            
            if not os.path.exists(dataset_path):
                raise ValueError(f"LiFE App dataset not found: {dataset_path}")
            
            with open(dataset_path, 'r') as f:
                life_app_data = json.load(f)
            
            # Create temporary directory for audio files
            temp_audio_dir = "./temp_audio_files"
            os.makedirs(temp_audio_dir, exist_ok=True)
            
            # Process each item
            dataset_rows = []
            for i, item in enumerate(life_app_data):
                # Get audio bytes directly from audio field
                audio_bytes = item["audio"]
                audio_filename = os.path.join(temp_audio_dir, f"audio_{i}.wav")
                
                # Save audio bytes to WAV file
                with open(audio_filename, 'wb') as f:
                    f.write(audio_bytes)
                
                dataset_rows.append({
                    "audio": audio_filename,
                    "transcription": item["transcription"]
                })
            
            # Create dataset
            from datasets import Dataset
            dataset = Dataset.from_list(dataset_rows)
            
            # Clean up temporary files
            for filename in os.listdir(temp_audio_dir):
                os.remove(os.path.join(temp_audio_dir, filename))
            os.rmdir(temp_audio_dir)
            
            return dataset
            
        except Exception as e:
            self.training_logger.error(f"Error loading dataset: {str(e)}")
            raise
            
    def _load_model_and_processor(self):
        """Load model and processor."""
        try:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                self.params.model,
                token=self.params.token if self.params.token else None,
                trust_remote_code=True
            )
            
            # Load model
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.params.model,
                    token=self.params.token if self.params.token else None,
                    trust_remote_code=True
                )
            except Exception:
                model = AutoModelForCTC.from_pretrained(
                    self.params.model,
                    token=self.params.token if self.params.token else None,
                    trust_remote_code=True
                )
            
            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            
            return model, processor
            
        except Exception as e:
            self.training_logger.error(f"Error loading model and processor: {str(e)}")
            raise
            
    def _train_model(self, model, processor, train_dataset, valid_dataset):
        """Train the model."""
        try:
            from transformers import Trainer, TrainingArguments
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.params.output_dir,
                per_device_train_batch_size=self.params.batch_size,
                per_device_eval_batch_size=self.params.batch_size,
                gradient_accumulation_steps=self.params.gradient_accumulation,
                learning_rate=self.params.lr,
                num_train_epochs=self.params.epochs,
                evaluation_strategy="epoch" if valid_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if valid_dataset else False,
                metric_for_best_model="wer" if valid_dataset else None,
                greater_is_better=False if valid_dataset else None,
                push_to_hub=self.params.push_to_hub,
                hub_model_id=self.params.hub_model_id,
                hub_token=self.params.token,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
            )
            
            # Train model
            trainer.train()
            
            # Save model
            trainer.save_model()
            
        except Exception as e:
            self.training_logger.error(f"Error training model: {str(e)}")
            raise 