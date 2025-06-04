import csv
import json
import librosa
import os
import numpy as np
from tqdm import tqdm

def csv_to_json(csv_path, json_path, audio_folder):
    data = []
    print(f"Reading CSV from: {csv_path}")
    print(f"Audio files from: {audio_folder}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for _ in reader)
        f.seek(0)
        next(reader)  # Skip header
        
        for row in tqdm(reader, total=total_rows, desc="Converting audio files"):
            try:
                audio_path = os.path.join(audio_folder, row['audio'])
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue
                    
                # Load audio file and convert to array
                array, sr = librosa.load(audio_path, sr=16000)
                
                # Convert to float32 to ensure compatibility
                array = array.astype(np.float32)
                
                data.append({
                    "audio": {
                        "array": array.tolist(),
                        "sampling_rate": sr
                    },
                    "transcription": row["transcription"]
                })
            except Exception as e:
                print(f"Error processing {row['audio']}: {str(e)}")
                continue
    
    print(f"Saving JSON to: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Conversion complete! {len(data)} files processed.")

# Convert training data
csv_to_json(
    r"C:\Users\Aryan\Downloads\archive\dataset.csv",
    "asr_training/train.json",
    r"C:\Users\Aryan\Downloads\archive\audio"
)

# Create training config
config = {
    "model_name": "facebook/wav2vec2-base-960h",
    "train_data": "asr_training/train.json",
    "eval_data": "asr_training/train.json",  # Using same data for eval
    "project_name": "asr-training",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 5e-5,
    "max_steps": -1,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": False,
    "fp16": True,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "save_total_limit": 1,
    "output_dir": "asr_training/output",
    "push_to_hub": False
}

print("Creating training config...")
with open("asr_training/training_config.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)
print("Training config created!") 