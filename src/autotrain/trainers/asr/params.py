# from typing import Optional
# from pydantic import Field
# from autotrain.trainers.common import AutoTrainParams

# class ASRParams(AutoTrainParams):
#     """
#     Parameters for Automatic Speech Recognition tasks.
    
#     Attributes:
#         task: The type of task, set to 'automatic-speech-recognition'.
#         base_model: The pre-trained model to use for fine-tuning.
#         project_name: Name of the project.
#         backend: Backend to use for training (e.g., 'local').
#         push_to_hub: Whether to push the model to Hugging Face Hub after training.
#         hub_token: Hugging Face authentication token.
#         hub_username: Hugging Face username.
#         data_path: Path to the dataset.
#         train_split: Training split of the dataset.
#         valid_split: Validation split of the dataset.
#         audio_column: Column containing audio data paths.
#         text_column: Column containing transcriptions.
#         epochs: Number of training epochs.
#         batch_size: Batch size per device.
#         lr: Learning rate.
#         optimizer: Optimizer to use (e.g., 'adamw_torch').
#         scheduler: Learning rate scheduler to use (e.g., 'linear').
#         mixed_precision: Mixed precision training (e.g., 'fp16').
#         gradient_accumulation: Number of gradient accumulation steps.
#         warmup_steps: Number of warmup steps for learning rate scheduling.
#         max_steps: Maximum number of training steps.
#         per_device_train_batch_size: Batch size per device for training.
#         per_device_eval_batch_size: Batch size per device for evaluation.
#         eval_strategy: Evaluation strategy (e.g., 'steps').
#         save_steps: Number of steps between model saves.
#         eval_steps: Number of steps between evaluations.
#         logging_steps: Number of steps between logging.
#         load_best_model_at_end: Whether to load the best model at the end of training.
#         metric_for_best_model: Metric to use for determining the best model (e.g., 'wer').
#         greater_is_better: Whether higher metric values are better.
#         group_by_length: Whether to group samples by audio length for batching.
#         fp16: Whether to use fp16 precision.
#         gradient_checkpointing: Whether to use gradient checkpointing.
#         save_total_limit: Maximum number of model checkpoints to keep.
#     """
    
#     task: str = Field("automatic-speech-recognition", title="Task")
#     base_model: str = Field("facebook/wav2vec2-large-960h", title="Base model")
#     project_name: str = Field("autotrain-asr", title="Project name")
#     backend: str = Field("local", title="Backend")
#     push_to_hub: bool = Field(True, title="Push to hub")
#     hub_token: Optional[str] = Field(None, title="Hub token")
#     hub_username: Optional[str] = Field(None, title="Hub username")
#     data_path: str = Field("", title="Data path")
#     train_split: str = Field("train", title="Train split")
#     valid_split: Optional[str] = Field("validation", title="Validation split")
#     audio_column: str = Field("audio", title="Audio column")
#     text_column: str = Field("transcription", title="Text column")
#     epochs: int = Field(3, title="Number of training epochs")
#     batch_size: int = Field(8, title="Batch size")
#     lr: float = Field(1e-5, title="Learning rate")
#     optimizer: str = Field("adamw_torch", title="Optimizer")
#     scheduler: str = Field("linear", title="Scheduler")
#     mixed_precision: str = Field("fp16", title="Mixed precision")
#     gradient_accumulation: int = Field(4, title="Gradient accumulation steps")
#     warmup_steps: int = Field(500, title="Warmup steps")
#     max_steps: int = Field(2000, title="Max steps")
#     per_device_train_batch_size: int = Field(8, title="Per device train batch size")
#     per_device_eval_batch_size: int = Field(8, title="Per device eval batch size")
#     eval_strategy: str = Field("steps", title="Evaluation strategy")
#     save_steps: int = Field(1000, title="Save steps")
#     eval_steps: int = Field(1000, title="Evaluation steps")
#     logging_steps: int = Field(25, title="Logging steps")
#     load_best_model_at_end: bool = Field(True, title="Load best model at end")
#     metric_for_best_model: str = Field("wer", title="Metric for best model")
#     greater_is_better: bool = Field(False, title="Greater is better")
#     group_by_length: bool = Field(True, title="Group by length")
#     fp16: bool = Field(True, title="FP16")
#     gradient_checkpointing: bool = Field(True, title="Gradient checkpointing")
#     save_total_limit: int = Field(3, title="Save total limit")




from typing import Optional
from pydantic import Field
from autotrain.trainers.common import AutoTrainParams

class ASRParams(AutoTrainParams):
    """
    Parameters for Automatic Speech Recognition tasks.
    """
    task: str = Field("asr", title="Task")
    model: str = Field("facebook/wav2vec2-base-960h", title="Base model")
    project_name: str = Field("autotrain-asr", title="Project name")
    backend: str = Field("local", title="Backend")
    push_to_hub: bool = Field(True, title="Push to hub")
    token: Optional[str] = Field(None, title="Hub Token")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    data_path: str = Field("", title="Data path")
    train_split: str = Field("train", title="Train split")
    valid_split: Optional[str] = Field("validation", title="Validation split")
    audio_column: str = Field("path", title="Audio column")
    text_column: str = Field("sentence", title="Text column")
    sampling_rate: int = Field(16000, title="Sampling rate")
    epochs: int = Field(3, title="Number of training epochs")
    batch_size: int = Field(8, title="Batch size")
    lr: float = Field(5e-5, title="Learning rate")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    mixed_precision: str = Field("fp16", title="Mixed precision")
    gradient_accumulation: int = Field(4, title="Gradient accumulation steps")
    warmup_steps: int = Field(500, title="Warmup steps")
    max_steps: int = Field(2000, title="Max steps")
    per_device_train_batch_size: int = Field(8, title="Per device train batch size")
    per_device_eval_batch_size: int = Field(8, title="Per device eval batch size")
    eval_strategy: str = Field("steps", title="Evaluation strategy")
    save_steps: int = Field(1000, title="Save steps")
    eval_steps: int = Field(1000, title="Evaluation steps")
    logging_steps: int = Field(25, title="Logging steps")
    load_best_model_at_end: bool = Field(True, title="Load best model at end")
    metric_for_best_model: str = Field("wer", title="Metric for best model")
    greater_is_better: bool = Field(False, title="Greater is better")
    group_by_length: bool = Field(True, title="Group by length")
    fp16: bool = Field(True, title="FP16")
    gradient_checkpointing: bool = Field(True, title="Gradient checkpointing")
    save_total_limit: int = Field(3, title="Save total limit")
    log: str = Field("tensorboard", title="Log")

    class Config:
        arbitrary_types_allowed = True

# from typing import Optional
# from pydantic import Field
# from autotrain.trainers.common import AutoTrainParams

# class ASRParams(AutoTrainParams):
#     """
#     Parameters for Automatic Speech Recognition tasks.
#     """
#     task: str = Field("automatic-speech-recognition", title="Task")
#     base_model: str = Field("facebook/wav2vec2-large-960h", title="Base model")
#     project_name: str = Field("autotrain-asr", title="Project name")
#     backend: str = Field("local", title="Backend")
#     push_to_hub: bool = Field(True, title="Push to hub")
#     hub_token: Optional[str] = Field(None, title="Hub token")
#     hub_username: Optional[str] = Field(None, title="Hub username")
#     data_path: str = Field("", title="Data path")
#     train_split: str = Field("train", title="Train split")
#     valid_split: Optional[str] = Field("validation", title="Validation split")
#     audio_column: str = Field("audio", title="Audio column")
#     text_column: str = Field("transcription", title="Text column")
#     epochs: int = Field(3, title="Number of training epochs")
#     batch_size: int = Field(8, title="Batch size")
#     lr: float = Field(1e-5, title="Learning rate")
#     optimizer: str = Field("adamw_torch", title="Optimizer")
#     scheduler: str = Field("linear", title="Scheduler")
#     mixed_precision: str = Field("fp16", title="Mixed precision")
#     gradient_accumulation: int = Field(4, title="Gradient accumulation steps")
#     warmup_steps: int = Field(500, title="Warmup steps")
#     max_steps: int = Field(2000, title="Max steps")
#     per_device_train_batch_size: int = Field(8, title="Per device train batch size")
#     per_device_eval_batch_size: int = Field(8, title="Per device eval batch size")
#     eval_strategy: str = Field("steps", title="Evaluation strategy")
#     save_steps: int = Field(1000, title="Save steps")
#     eval_steps: int = Field(1000, title="Evaluation steps")
#     logging_steps: int = Field(25, title="Logging steps")
#     load_best_model_at_end: bool = Field(True, title="Load best model at end")
#     metric_for_best_model: str = Field("wer", title="Metric for best model")
#     greater_is_better: bool = Field(False, title="Greater is better")
#     group_by_length: bool = Field(True, title="Group by length")
#     fp16: bool = Field(True, title="FP16")
#     gradient_checkpointing: bool = Field(True, title="Gradient checkpointing")
#     save_total_limit: int = Field(3, title="Save total limit")
#     log: str = Field("tensorboard", title="Log")