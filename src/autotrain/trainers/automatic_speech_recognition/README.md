# AutoTrain Advanced: Automatic Speech Recognition (ASR) Task

## Overview
This module provides full support for Automatic Speech Recognition (ASR) in AutoTrain Advanced, including:
- Training with local datasets (audio + CSV/JSON)
- Training with Hugging Face Hub datasets
- Seamless integration with LiFE App datasets
- Consistent workflow, UI, and code hygiene as other tasks (image, text, etc.)

## File Structure
```
src/autotrain/trainers/automatic_speech_recognition/
    __main__.py         # Training entry point and pipeline
    dataset.py          # Universal ASR dataset class
    utils.py            # Metrics, model card, callbacks
    params.py           # Training parameter config
src/autotrain/preprocessor/automatic_speech_recognition.py  # Local/LiFE App dataset validation/prep
```

## Supported Dataset Sources
- **Local**: Upload a zip with an audio folder and a CSV/JSON file (columns: `audio`, `transcription`).
- **Hugging Face Hub**: Enter dataset name, splits, and column mapping.
- **LiFE App**: (ASR only) Select a LiFE App project/script; backend converts to local format automatically.

### Local Dataset Format
- Folder must contain:
  - `audio/` directory with audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`)
  - CSV or JSON file with columns:
    - `audio`: filename (not path)
    - `transcription`: ground truth text

### LiFE App Integration
- LiFE App datasets are converted to the above local format automatically.
- All validation, mapping, and error handling is shared with other sources.

## Training Parameters
All standard parameters are supported (see `params.py`):
- `model`: Pretrained model name or path (e.g., `facebook/wav2vec2-base-960h`)
- `batch_size`, `epochs`, `lr`, `warmup_ratio`, `weight_decay`, etc.
- `audio_column`, `text_column`, `max_duration`, `sampling_rate`, `max_seq_length`
- `push_to_hub`, `logging_steps`, `early_stopping_patience`, etc.

## Usage
### 1. From the UI
- Select **ASR** as the task.
- Choose dataset source (Local, Hub, or LiFE App).
- Map columns as needed.
- Set parameters (basic or full).
- Start training. Progress and logs are shown in the UI.

### 2. From the CLI (example)
```bash
python -m autotrain.trainers.automatic_speech_recognition --training_config path/to/config.json
```

### 3. Training Config Example
```json
{
  "data_path": "my_project/autotrain-data",
  "model": "openai/whisper-small",
  "batch_size": 8,
  "epochs": 3,
  "audio_column": "audio",
  "text_column": "transcription",
  "sampling_rate": 16000,
  "max_duration": 30.0
}
```

## Metrics & Model Card
- WER (Word Error Rate), CER (Character Error Rate), and accuracy are computed and logged.
- A model card is generated automatically after training, including validation metrics.

## Code Hygiene & Extensibility
- All logic is modular, main-repo style, and well-documented.
- No dead or duplicate code; all common logic is shared with other tasks.
- To add new models or dataset formats, extend the relevant helpers in `dataset.py` and `preprocessor/automatic_speech_recognition.py`.

## Advanced Usage
- **Custom Models:**
  - Add new model support by updating `detect_model_type` in `dataset.py` and model loading logic in `__main__.py`.
- **Custom Metrics:**
  - Extend or replace `compute_metrics` in `utils.py`.
- **Custom Callbacks:**
  - Add new callbacks in `utils.py` and register them in the `train` function in `__main__.py`.

## Example Output
**Training Log Snippet:**
```
[INFO] Using device: cuda
[INFO] Training dataset object created with 1200 examples.
[INFO] TRAINING STARTED - Watch for progress logs
[INFO] TRAINING COMPLETED
[INFO] Final evaluation results: {'eval_loss': 0.32, 'eval_wer': 0.12, 'eval_cer': 0.05, 'eval_accuracy': 0.88}
```

**Model Card Snippet:**
```
## Validation Metrics
wer: 0.12

cer: 0.05

accuracy: 0.88
```

## ASR Training Pipeline (Diagram)
```mermaid
graph TD;
  A[Start: Select ASR Task] --> B{Choose Dataset Source};
  B -->|Local| C[Upload audio + CSV/JSON];
  B -->|Hugging Face Hub| D[Enter dataset name/split];
  B -->|LiFE App| E[Select project/script];
  C & D & E --> F[Preprocessing & Validation];
  F --> G[Model/Processor Loading];
  G --> H[Training Loop];
  H --> I[Evaluation & Metrics];
  I --> J[Model Card & Save];
  J --> K[Push to Hub (optional)];
```

## Further Reading & Support
- [AutoTrain Advanced Main Repo](https://github.com/huggingface/autotrain-advanced)
- [AutoTrain Advanced Documentation](https://huggingface.co/docs/autotrain)
- [Open an Issue](https://github.com/huggingface/autotrain-advanced/issues)

## Work Done / Improvements
- Fixed the previous issue where users would see repeated or false 'token verification failed' errors. Authentication logic is now robust and user-friendly, ensuring smooth login and operation for both UI and backend. 