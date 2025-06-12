import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset_json_path, output_dir):
    """
    Prepare the LiFE App Dataset for ASR training.
    
    Args:
        dataset_json_path: Path to the dataset.json file
        output_dir: Directory to save the processed dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Split into train and validation sets
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save train and validation sets
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    
    print(f"Dataset prepared and saved to {output_dir}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")

def main():
    # Configuration
    dataset_json_path = "dataset.json"  # Path to your dataset.json file
    output_dir = "data"  # Directory to save processed dataset
    
    # Prepare dataset
    prepare_dataset(dataset_json_path, output_dir)
    
    # Print instructions for training
    print("\nTo start training, run:")
    print("autotrain --config configs/automatic_speech_recognition/life_app_dataset.yml")

if __name__ == "__main__":
    main() 