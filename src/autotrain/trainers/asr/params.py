from typing import Optional
from pydantic import Field
from autotrain.trainers.common import AutoTrainParams


class ASRParams(AutoTrainParams):
    """
    ASRParams is a configuration class for Automatic Speech Recognition (ASR) training parameters.

    Attributes:
        data_path (str): Path to the dataset.
        model (str): Name of the pre-trained model to use. Default is "facebook/wav2vec2-base-960h".
        lr (float): Learning rate for the optimizer. Default is 3e-4.
        epochs (int): Number of training epochs. Default is 3.
        batch_size (int): Batch size for training. Default is 8.
        warmup_steps (int): Number of warmup steps for learning rate scheduler. Default is 500.
        gradient_accumulation (int): Number of steps to accumulate gradients before updating. Default is 2.
        save_steps (int): Number of steps between saving checkpoints. Default is 500.
        eval_steps (int): Number of steps between evaluations. Default is 500.
        project_name: str = Field("project-name", title="Name of the project for output directory")
        train_split: str = Field("train", title="Name of the training data split")
        valid_split: Optional[str] = Field(None, title="Name of the validation data split")
        username: Optional[str] = Field(None, title="Hugging Face username for authentication")
        logging_steps (int): Number of steps between logging. Default is 50.
        audio_column (str): Name of the column containing audio data. Default is "audio".
        text_column (str): Name of the column containing text data. Default is "text".
        lr: float = Field(5e-5, title="Learning rate for the optimizer")
        warmup_ratio: float = Field(0.1, title="Warmup ratio for learning rate scheduler")
        optimizer: str = Field("adamw_torch", title="Optimizer type")
        scheduler: str = Field("linear", title="Learning rate scheduler type")
        weight_decay: float = Field(0.0, title="Weight decay for the optimizer")
        max_grad_norm: float = Field(1.0, title="Maximum gradient norm for clipping")
        seed: int = Field(42, title="Random seed for reproducibility")
        auto_find_batch_size: bool = Field(False, title="Automatically find optimal batch size")
        mixed_precision: Optional[str] = Field(None, title="Mixed precision training mode (fp16, bf16, or None)")
        save_total_limit: int = Field(1, title="Maximum number of checkpoints to keep")
        token: Optional[str] = Field(None, title="Hugging Face Hub token for authentication")
        push_to_hub: bool = Field(False, title="Whether to push the model to Hugging Face Hub")
        eval_strategy: str = Field("epoch", title="Evaluation strategy during training")
        image_column: str = Field("image", title="Column name for images in the dataset")
        target_column: str = Field("target", title="Column name for target labels in the dataset")
        log: str = Field("none", title="Logging method for experiment tracking")
        early_stopping_patience: int = Field(5, title="Number of epochs with no improvement for early stopping")
        early_stopping_threshold: float = Field(0.01, title="Threshold for early stopping")
    """

    data_path: str = Field(None, title="Data path")
    model: str = Field("facebook/wav2vec2-base-960h", title="Model name")
    lr: float = Field(3e-4, title="Learning rate")
    epochs: int = Field(10, title="Number of training epochs")
    batch_size: int = Field(16, title="Batch size for training")
    warmup_steps: int = Field(500, title="Warmup steps")
    project_name: str = Field("project-name", title="Name of the project for output directory")
    gradient_accumulation: int = Field(2, title="Gradient accumulation steps")
    save_steps: int = Field(500, title="Save steps")
    eval_steps: int = Field(500, title="Evaluation steps")
    logging_steps: int = Field(50, title="Logging steps")
    train_split: str = Field("train", title="Name of the training data split")
    valid_split: Optional[str] = Field(None, title="Name of the validation data split")
    username: Optional[str] = Field(None, title="Hugging Face username for authentication")
    token: Optional[str] = Field(None, title="Authentication token for Hugging Face Hub")
    audio_column: str = Field("audio", title="Audio column")
    log: str = Field("none", title="Logging method for experiment tracking")
    text_column: str = Field("text", title="Text column")
    eval_strategy: str = Field("steps", title="Evaluation strategy")
    save_total_limit: int = Field(1, title="Save total limit")
    push_to_hub: bool = Field(False, title="Push to hub")
    lr: float = Field(5e-5, title="Learning rate for the optimizer")
    epochs: int = Field(3, title="Number of epochs for training")
    batch_size: int = Field(8, title="Batch size for training")
    warmup_ratio: float = Field(0.1, title="Warmup ratio for learning rate scheduler")
    gradient_accumulation: int = Field(1, title="Number of gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer type")
    scheduler: str = Field("linear", title="Learning rate scheduler type")
    weight_decay: float = Field(0.0, title="Weight decay for the optimizer")
    max_grad_norm: float = Field(1.0, title="Maximum gradient norm for clipping")
    seed: int = Field(42, title="Random seed for reproducibility")
    train_split: str = Field("train", title="Name of the training data split")
    valid_split: Optional[str] = Field(None, title="Name of the validation data split")
    logging_steps: int = Field(-1, title="Number of steps between logging")
    project_name: str = Field("project-name", title="Name of the project for output directory")
    auto_find_batch_size: bool = Field(False, title="Automatically find optimal batch size")
    mixed_precision: Optional[str] = Field(None, title="Mixed precision training mode (fp16, bf16, or None)")
    save_total_limit: int = Field(1, title="Maximum number of checkpoints to keep")
    token: Optional[str] = Field(None, title="Hugging Face Hub token for authentication")
    push_to_hub: bool = Field(False, title="Whether to push the model to Hugging Face Hub")
    eval_strategy: str = Field("epoch", title="Evaluation strategy during training")
    image_column: str = Field("image", title="Column name for images in the dataset")
    target_column: str = Field("target", title="Column name for target labels in the dataset")
    log: str = Field("none", title="Logging method for experiment tracking")
    early_stopping_patience: int = Field(5, title="Number of epochs with no improvement for early stopping")
    early_stopping_threshold: float = Field(0.01, title="Threshold for early stopping")