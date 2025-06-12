import os
import logging
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path

from autotrain.preprocessor.automatic_speech_recognition import AutomaticSpeechRecognitionPreprocessor
from .data_fetcher import LifeAppDataFetcher

logger = logging.getLogger(__name__)

class LifeAppASRPreprocessor(AutomaticSpeechRecognitionPreprocessor):
    """
    Preprocessor for LiFE App ASR data.
    Extends the base ASR preprocessor with LiFE App specific functionality.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        json_file_path: Optional[str] = None,
        project_name: str = "life-app-asr",
        username: str = "",
        token: str = "",
        column_mapping: Optional[Dict[str, str]] = None,
        test_size: float = 0.2,
        seed: int = 42,
        local: bool = False,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            api_url: URL for LiFE App API (optional)
            api_token: Authentication token for API (optional)
            json_file_path: Path to local JSON file (optional)
            project_name: Name of the project
            username: Hugging Face username
            token: Hugging Face token
            column_mapping: Mapping of column names
            test_size: Proportion of data to use for validation
            seed: Random seed for splitting
            local: Whether to save data locally
        """
        # Initialize data fetcher
        self.data_fetcher = LifeAppDataFetcher(
            api_url=api_url,
            api_token=api_token,
            json_file_path=json_file_path
        )
        
        # Fetch and prepare data
        data = self.data_fetcher.fetch_data()
        train_df = pd.DataFrame(data)
        
        # Set default column mapping if not provided
        if column_mapping is None:
            column_mapping = {
                "audio": "audio",
                "transcription": "transcription",
                "duration": "duration"
            }
        
        # Initialize parent class
        super().__init__(
            train_data=train_df,
            project_name=project_name,
            username=username,
            token=token,
            column_mapping=column_mapping,
            test_size=test_size,
            seed=seed,
            local=local,
        )
        
    def prepare(self) -> str:
        """
        Prepare the dataset for training.
        
        Returns:
            Path to the prepared dataset
        """
        logger.info("Preparing LiFE App ASR dataset...")
        
        # Process the data
        processed_df = self._process_data(self.train_data)
        
        # Create datasets
        datasets = self._create_datasets(processed_df)
        
        # Save datasets
        output_dir = os.path.join(self.project_name, "data")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train and validation sets
        train_path = os.path.join(output_dir, "train.csv")
        valid_path = os.path.join(output_dir, "valid.csv")
        
        datasets["train"].to_csv(train_path, index=False)
        if "validation" in datasets:
            datasets["validation"].to_csv(valid_path, index=False)
            
        logger.info(f"Dataset prepared and saved to {output_dir}")
        return output_dir 