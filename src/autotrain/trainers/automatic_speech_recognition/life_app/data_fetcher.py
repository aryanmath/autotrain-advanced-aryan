import os
import json
import logging
import requests
from typing import Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class LifeAppDataFetcher:
    """
    Data fetcher for LiFE App ASR dataset.
    Handles both local JSON file and API data fetching.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        json_file_path: Optional[str] = None
    ):
        """
        Initialize the data fetcher.
        
        Args:
            api_url: URL for LiFE App API (optional)
            api_token: Authentication token for API (optional)
            json_file_path: Path to local JSON file (optional)
        """
        self.api_url = api_url
        self.api_token = api_token
        self.json_file_path = json_file_path
        
        # Validate inputs
        if not api_url and not json_file_path:
            raise ValueError("Either api_url or json_file_path must be provided")
            
    def fetch_data(self) -> List[Dict]:
        """
        Fetch data from either API or local JSON file.
        
        Returns:
            List of dictionaries containing audio and transcription data
        """
        if self.api_url:
            return self._fetch_from_api()
        else:
            return self._fetch_from_json()
            
    def _fetch_from_api(self) -> List[Dict]:
        """Fetch data from LiFE App API."""
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
            response = requests.get(self.api_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return self._validate_and_transform_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from API: {str(e)}")
            raise
            
    def _fetch_from_json(self) -> List[Dict]:
        """Fetch data from local JSON file."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._validate_and_transform_data(data)
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error reading JSON file: {str(e)}")
            raise
            
    def _validate_and_transform_data(self, data: Union[List, Dict]) -> List[Dict]:
        """
        Validate and transform data into required format.
        
        Args:
            data: Raw data from API or JSON file
            
        Returns:
            List of validated and transformed data items
        """
        if isinstance(data, dict):
            data = [data]
            
        validated_data = []
        for item in data:
            # Validate required fields
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid item: {item}")
                continue
                
            if 'audio' not in item or 'transcription' not in item:
                logger.warning(f"Skipping item missing required fields: {item}")
                continue
                
            # Transform data if needed
            transformed_item = {
                'audio': item['audio'],
                'transcription': item['transcription']
            }
            
            # Add optional fields if present
            if 'duration' in item:
                transformed_item['duration'] = item['duration']
                
            validated_data.append(transformed_item)
            
        return validated_data 