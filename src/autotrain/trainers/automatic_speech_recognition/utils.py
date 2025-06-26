import json
from typing import Dict, Any
import numpy as np
from transformers import Trainer
import jiwer

# At the top of your file, set processor as a global variable
processor = None

def set_processor(proc):
    global processor
    processor = proc

def compute_metrics(pred):
    import jiwer
    import numpy as np

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    # Decode
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute metrics
    wer = jiwer.wer(label_str, pred_str)
    cer = jiwer.cer(label_str, pred_str)

    return {"wer": wer, "cer": cer}

def create_model_card(config: Dict[str, Any], trainer: Trainer, num_classes: int = None) -> str:
    """Create a model card for the trained ASR model."""
    model_card = f"""# {config.project_name}

This model was trained using AutoTrain.

## Training Configuration

- Base Model: {config.model}
- Task: Automatic Speech Recognition
- Training Data: {config.data_path}
- Validation Data: {config.valid_split if config.valid_split else "None"}
- Epochs: {config.epochs}
- Batch Size: {config.batch_size}
- Learning Rate: {config.lr}
- Optimizer: {config.optimizer}
- Scheduler: {config.scheduler}
- Mixed Precision: {config.mixed_precision}

## Training Results

- Final Loss: {trainer.state.log_history[-1]['loss'] if trainer.state.log_history else "N/A"}
- Best Validation Loss: {trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else "N/A"}

## Usage

```python
from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch
import librosa

# Load model and processor
model = AutoModelForCTC.from_pretrained("{config.username}/{config.project_name}")
processor = Wav2Vec2Processor.from_pretrained("{config.username}/{config.project_name}")

# Load and preprocess audio
audio, sr = librosa.load("path_to_audio.wav", sr={config.sampling_rate})
inputs = processor(audio, sampling_rate={config.sampling_rate}, return_tensors="pt", padding=True)

# Get predictions
with torch.no_grad():
    logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
```
"""
    return model_card 