import os
import requests
import pandas as pd
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class LifeAppClient:
    def __init__(self, api_key: str):
        """Initialize the Life App client with API key."""
        self.api_key = api_key
        self.base_url = "https://api.lifeapp.com/v1"  # Replace with actual Life App API URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def get_dataset(self, dataset_id: str) -> dict:
        """Get dataset metadata from Life App."""
        try:
            response = requests.get(
                f"{self.base_url}/datasets/{dataset_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching dataset metadata: {str(e)}")
            raise

    def get_dataset_files(self, dataset_id: str) -> dict:
        """Get files associated with the dataset."""
        try:
            response = requests.get(
                f"{self.base_url}/datasets/{dataset_id}/files",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching dataset files: {str(e)}")
            raise

    def download_file(self, file_url: str, local_path: str) -> None:
        """Download a file from Life App."""
        try:
            response = requests.get(file_url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise

    def prepare_dataset(self, dataset_id: str, output_dir: str) -> Dataset:
        """Prepare the dataset for training by downloading and organizing files."""
        try:
            # Get dataset metadata
            dataset_info = self.get_dataset(dataset_id)
            files_info = self.get_dataset_files(dataset_id)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            audio_dir = os.path.join(output_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)

            # Download and organize files
            data = []
            for file_info in files_info['files']:
                if file_info['type'] == 'audio':
                    # Download audio file
                    local_path = os.path.join(audio_dir, file_info['filename'])
                    self.download_file(file_info['url'], local_path)
                    
                    # Add to dataset
                    data.append({
                        'audio': local_path,
                        'transcription': file_info['transcription']
                    })

            # Create DataFrame and save as CSV
            df = pd.DataFrame(data)
            csv_path = os.path.join(output_dir, "dataset.csv")
            df.to_csv(csv_path, index=False)

            # Convert to HuggingFace Dataset
            dataset = Dataset.from_pandas(df)
            return dataset

        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise 