"""
This file demonstrates how LiFE App ASR code integrates with existing AutoTrain code.
"""

# 1. Integration with ASR Task Files
from autotrain.trainers.automatic_speech_recognition.__main__ import train
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
from autotrain.trainers.automatic_speech_recognition.dataset import AutomaticSpeechRecognitionDataset
from autotrain.trainers.automatic_speech_recognition.utils import compute_metrics

# 2. Integration with Preprocessor
from autotrain.preprocessor.automatic_speech_recognition import AutomaticSpeechRecognitionPreprocessor

# 3. Integration with CLI
from autotrain.cli.run_automatic_speech_recognition import RunAutoTrainAutomaticSpeechRecognitionCommand

# 4. Integration with Backend
from autotrain.backends.base import BaseBackend
from autotrain.backends.local import LocalRunner
from autotrain.backends.asr_backend import ASRBackend

# 5. Integration with App
from autotrain.app.db import AutoTrainDB
from autotrain.app.models import fetch_models

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from autotrain import logger
from .trainer import LifeAppASRTrainer

# Example of how LiFE App code uses these integrations:

class LifeAppASRIntegration:
    """
    Integration class for LiFE App ASR training.
    Connects UI to training code.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the integration.
        
        Args:
            config: Training configuration from UI
        """
        self.config = config
        self.params = AutomaticSpeechRecognitionParams(**config)
        self.trainer = LifeAppASRTrainer(config)
        
    def start_training(self):
        """Start ASR training with selected model."""
        try:
            logger.info(f"Starting ASR training with model: {self.params.model}")
            logger.info(f"Using LiFE App dataset: {self.params.life_app_dataset_name}")
            
            # Start training
            self.trainer.train()
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    @staticmethod
    def get_available_models() -> list:
        """Get list of available ASR models."""
        try:
            from autotrain.app.models import fetch_models
            models = fetch_models()
            return models.get("automatic-speech-recognition", [])
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            return []
            
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate training configuration."""
        try:
            # Check required fields
            required_fields = [
                "project_name",
                "model",
                "life_app_dataset_name",
                "username",
                "token"
            ]
            
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Validate model exists
            models = LifeAppASRIntegration.get_available_models()
            if config["model"] not in models:
                raise ValueError(f"Model {config['model']} not found in available models")
                
            # Validate dataset exists
            dataset_path = os.path.join("src/autotrain/app/static", config["life_app_dataset_name"])
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset not found: {dataset_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating config: {str(e)}")
            return False 