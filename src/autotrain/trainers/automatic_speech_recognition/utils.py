import json
from typing import Dict, Any
import numpy as np
from transformers import Trainer
import jiwer
import os


processor = None
#hello
def set_processor(proc):
    global processor
    processor = proc

def compute_metrics(pred):
    import jiwer
    import numpy as np

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]

    
    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)

   
    if pred_ids.dtype != np.int32 and pred_ids.dtype != np.int64:
        pred_ids = np.argmax(pred_ids, axis=-1)

    
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    
    pred_ids = pred_ids.astype(int).tolist()
    label_ids = label_ids.astype(int).tolist()

    
    if isinstance(pred_ids[0], list):
        pass  
    else:
        pred_ids = [pred_ids]
    if isinstance(label_ids[0], list):
        pass
    else:
        label_ids = [label_ids]

    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    
    wer = jiwer.wer(label_str, pred_str)
    cer = jiwer.cer(label_str, pred_str)

    return {"wer": wer, "cer": cer}

MODEL_CARD = """
---
tags:
- autotrain
- transformers
- automatic-speech-recognition{base_model}
widget:
- src: https://huggingface.co/datasets/mishig/sample_audio/resolve/main/sample1.wav
  example_title: Sample Audio 1
- src: https://huggingface.co/datasets/mishig/sample_audio/resolve/main/sample2.wav
  example_title: Sample Audio 2{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Automatic Speech Recognition

## Validation Metrics
{validation_metrics}

## Training Configuration
- Base Model: {base_model}
- Training Data: {training_data}
- Validation Data: {validation_data}
- Epochs: {epochs}
- Batch Size: {batch_size}
- Learning Rate: {learning_rate}
- Optimizer: {optimizer}
- Scheduler: {scheduler}
- Mixed Precision: {mixed_precision}

## Usage
```python
from transformers import AutoModelForCTC, Wav2Vec2Processor
import torch
import librosa

# Load model and processor
model = AutoModelForCTC.from_pretrained("{username}/{project_name}")
processor = Wav2Vec2Processor.from_pretrained("{username}/{project_name}")

audio, sr = librosa.load("path_to_audio.wav", sr={sampling_rate})
inputs = processor(audio, sampling_rate={sampling_rate}, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
```
"""

def create_model_card(config: Dict[str, Any], trainer: Trainer, num_classes: int = None) -> str:
    """Create a detailed model card for the trained ASR model."""
    # Validation metrics (WER/CER)
    if hasattr(trainer, 'evaluate'):
        eval_scores = trainer.evaluate() if (hasattr(config, 'valid_split') and config.valid_split) else None
        if eval_scores:
            valid_metrics = [
                f"wer: {eval_scores.get('wer', 'N/A')}",
                f"cer: {eval_scores.get('cer', 'N/A')}"
            ]
            validation_metrics = "\n\n".join(valid_metrics)
        else:
            validation_metrics = "No validation metrics available"
    else:
        validation_metrics = "No validation metrics available"

    # Dataset tag
    if hasattr(config, 'data_path') and (config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path)):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {getattr(config, 'data_path', '')}"

    # Base model
    if hasattr(config, 'model') and os.path.isdir(config.model):
        base_model = ""
    else:
        base_model = f"\nbase_model: {getattr(config, 'model', '')}"

    # Fill in the template
    model_card = MODEL_CARD.format(
        base_model=base_model,
        dataset_tag=dataset_tag,
        validation_metrics=validation_metrics,
        training_data=getattr(config, 'data_path', ''),
        validation_data=getattr(config, 'valid_split', 'None'),
        epochs=getattr(config, 'epochs', 'N/A'),
        batch_size=getattr(config, 'batch_size', 'N/A'),
        learning_rate=getattr(config, 'lr', 'N/A'),
        optimizer=getattr(config, 'optimizer', 'N/A'),
        scheduler=getattr(config, 'scheduler', 'N/A'),
        mixed_precision=getattr(config, 'mixed_precision', 'N/A'),
        username=getattr(config, 'username', 'username'),
        project_name=getattr(config, 'project_name', 'project_name'),
        sampling_rate=getattr(config, 'sampling_rate', 16000),
    )
    return model_card 