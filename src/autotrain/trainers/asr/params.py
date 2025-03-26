from dataclasses import dataclass
from typing import Optional

from autotrain.trainers.common import BaseParams

@dataclass
class ASRParams(AutoTrainParams):
    """
    Parameters for Automatic Speech Recognition tasks.
    
    Attributes:
        task: The type of task, set to 'automatic-speech-recognition'.
        base_model: The pre-trained model to use for fine-tuning.
        project_name: Name of the project.
        log: Logging framework to use (e.g., 'tensorboard').
        backend: Backend to use for training (e.g., 'local').
        push_to_hub: Whether to push the model to Hugging Face Hub after training.
        hub_token: Hugging Face authentication token.
        hub_username: Hugging Face username.
        data_path: Path to the dataset.
        train_split: Training split of the dataset.
        valid_split: Validation split of the dataset.
        audio_column: Column containing audio data paths.
        text_column: Column containing transcriptions.
        epochs: Number of training epochs.
        batch_size: Batch size per device.
        lr: Learning rate.
        optimizer: Optimizer to use (e.g., 'adamw_torch').
        scheduler: Learning rate scheduler to use (e.g., 'linear').
        mixed_precision: Mixed precision training (e.g., 'fp16').
        gradient_accumulation: Number of gradient accumulation steps.
        warmup_steps: Number of warmup steps for learning rate scheduling.
        max_steps: Maximum number of training steps.
        per_device_train_batch_size: Batch size per device for training.
        per_device_eval_batch_size: Batch size per device for evaluation.
        eval_strategy: Evaluation strategy (e.g., 'steps').
        save_steps: Number of steps between model saves.
        eval_steps: Number of steps between evaluations.
        logging_steps: Number of steps between logging.
        load_best_model_at_end: Whether to load the best model at the end of training.
        metric_for_best_model: Metric to use for determining the best model (e.g., 'wer').
        greater_is_better: Whether higher metric values are better.
        group_by_length: Whether to group samples by audio length for batching.
        fp16: Whether to use fp16 precision.
        gradient_checkpointing: Whether to use gradient checkpointing.
        save_total_limit: Maximum number of model checkpoints to keep.
    """
    
    task: str = "automatic-speech-recognition"
    base_model: str = "facebook/wav2vec2-large-960h"
    project_name: str = "autotrain-asr"
    log: str = "tensorboard"
    backend: str = "local"
    push_to_hub: bool = True
    hub_token: Optional[str] = None
    hub_username: Optional[str] = None
    
    data_path: str = ""
    train_split: str = "train"
    valid_split: Optional[str] = "validation"
    audio_column: str = "audio"
    text_column: str = "transcription"
    
    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-5
    optimizer: str = "adamw_torch"
    scheduler: str = "linear"
    mixed_precision: str = "fp16"
    gradient_accumulation: int = 4
    warmup_steps: int = 500
    max_steps: int = 2000
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    eval_strategy: str = "steps"
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 25
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    group_by_length: bool = True
    fp16: bool = True
    gradient_checkpointing: bool = True
    save_total_limit: int = 3