import os
import logging
import json
from typing import Dict, Optional
from pathlib import Path

from autotrain.trainers.automatic_speech_recognition.__main__ import train
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
from .preprocessor import LifeAppASRPreprocessor

logger = logging.getLogger(__name__)

def train_life_app_asr(
    api_url: Optional[str] = None,
    api_token: Optional[str] = None,
    json_file_path: Optional[str] = None,
    project_name: str = "life-app-asr",
    username: str = "",
    token: str = "",
    model: str = "facebook/wav2vec2-base-960h",
    max_duration: float = 30.0,
    sampling_rate: int = 16000,
    batch_size: int = 8,
    epochs: int = 5,
    learning_rate: float = 3e-4,
    **kwargs
) -> str:
    """
    Train ASR model on LiFE App data.
    
    Args:
        api_url: URL for LiFE App API (optional)
        api_token: Authentication token for API (optional)
        json_file_path: Path to local JSON file (optional)
        project_name: Name of the project
        username: Hugging Face username
        token: Hugging Face token
        model: Base model to use
        max_duration: Maximum audio duration in seconds
        sampling_rate: Target sampling rate
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        **kwargs: Additional training parameters
        
    Returns:
        Path to the trained model
    """
    logger.info("Starting LiFE App ASR training...")
    
    # Initialize preprocessor
    preprocessor = LifeAppASRPreprocessor(
        api_url=api_url,
        api_token=api_token,
        json_file_path=json_file_path,
        project_name=project_name,
        username=username,
        token=token
    )
    
    # Prepare dataset
    data_path = preprocessor.prepare()
    
    # Create training parameters
    params = AutomaticSpeechRecognitionParams(
        project_name=project_name,
        data_path=data_path,
        model=model,
        username=username,
        token=token,
        max_duration=max_duration,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        **kwargs
    )
    
    # Start training
    train(params.model_dump())
    
    # Return path to trained model
    model_path = os.path.join(project_name, "model")
    logger.info(f"Training completed. Model saved to {model_path}")
    return model_path

def main():
    """Main function to run LiFE App ASR training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ASR model on LiFE App data")
    
    # Data source arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--api-url", help="URL for LiFE App API")
    data_group.add_argument("--json-file", help="Path to local JSON file")
    
    parser.add_argument("--api-token", help="Authentication token for API")
    
    # Training arguments
    parser.add_argument("--project-name", default="life-app-asr", help="Name of the project")
    parser.add_argument("--username", default="", help="Hugging Face username")
    parser.add_argument("--token", default="", help="Hugging Face token")
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h", help="Base model to use")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum audio duration in seconds")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Target sampling rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Start training
    train_life_app_asr(
        api_url=args.api_url,
        api_token=args.api_token,
        json_file_path=args.json_file,
        project_name=args.project_name,
        username=args.username,
        token=args.token,
        model=args.model,
        max_duration=args.max_duration,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main() 