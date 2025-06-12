# LiFE App ASR Training Setup

This setup allows you to train an Automatic Speech Recognition (ASR) model on the LiFE App Dataset using AutoTrain Advanced.

## Prerequisites

1. Python 3.8 or higher
2. Required Python packages:
   ```bash
   pip install pandas scikit-learn transformers datasets librosa soundfile
   ```

## Dataset Structure

The dataset should be in JSON format with the following structure:
```json
[
    {
        "audio": "path/to/audio/file.wav",
        "transcription": "text transcription of the audio"
    },
    ...
]
```

## Setup and Training

1. **Prepare the Dataset**:
   ```bash
   python train_life_app_asr.py
   ```
   This will:
   - Load your dataset.json file
   - Split it into training and validation sets
   - Save them as CSV files in the `data` directory

2. **Start Training**:
   ```bash
   autotrain --config configs/automatic_speech_recognition/life_app_dataset.yml
   ```

## Configuration

The training configuration is in `configs/automatic_speech_recognition/life_app_dataset.yml`. Key settings:

- `base_model`: Using facebook/wav2vec2-base-960h as the base model
- `max_duration`: 30 seconds (maximum audio length)
- `sampling_rate`: 16000 Hz
- `epochs`: 5
- `batch_size`: 8
- `learning_rate`: 3e-4

You can modify these settings in the YAML file based on your needs.

## Training Process

1. The training will:
   - Load and preprocess your audio files
   - Convert transcriptions to the required format
   - Train the model using the specified configuration
   - Save checkpoints and logs

2. Training progress can be monitored through:
   - Console output
   - TensorBoard logs (if enabled)

## Model Output

The trained model will be saved in the `autotrain-asr-life-app` directory. You can use it for inference or push it to the Hugging Face Hub by setting `push_to_hub: true` in the config file.

## Troubleshooting

1. If you encounter memory issues:
   - Reduce the batch size
   - Reduce max_duration
   - Use gradient accumulation

2. If training is too slow:
   - Enable mixed precision training (already enabled in config)
   - Use a smaller model
   - Reduce the number of epochs

3. If you get audio processing errors:
   - Ensure all audio files are in a supported format (WAV recommended)
   - Check if the audio paths in your dataset are correct
   - Verify the sampling rate of your audio files 