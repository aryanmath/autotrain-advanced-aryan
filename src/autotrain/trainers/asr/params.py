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
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Batch size for training. Default is 16.
        warmup_steps (int): Number of warmup steps for learning rate scheduler. Default is 500.
        gradient_accumulation (int): Number of steps to accumulate gradients before updating. Default is 2.
        save_steps (int): Number of steps between saving checkpoints. Default is 500.
        eval_steps (int): Number of steps between evaluations. Default is 500.
        logging_steps (int): Number of steps between logging. Default is 50.
        audio_column (str): Name of the column containing audio data. Default is "audio".
        text_column (str): Name of the column containing text data. Default is "text".
    """

    data_path: str = Field(None, title="Data path")
    model: str = Field("facebook/wav2vec2-base-960h", title="Model name")
    lr: float = Field(3e-4, title="Learning rate")
    epochs: int = Field(10, title="Number of training epochs")
    batch_size: int = Field(16, title="Batch size for training")
    warmup_steps: int = Field(500, title="Warmup steps")
    gradient_accumulation: int = Field(2, title="Gradient accumulation steps")
    save_steps: int = Field(500, title="Save steps")
    eval_steps: int = Field(500, title="Evaluation steps")
    logging_steps: int = Field(50, title="Logging steps")
    audio_column: str = Field("audio", title="Audio column")
    text_column: str = Field("text", title="Text column")
    eval_strategy: str = Field("steps", title="Evaluation strategy")
    save_total_limit: int = Field(1, title="Save total limit")
    push_to_hub: bool = Field(False, title="Push to hub")