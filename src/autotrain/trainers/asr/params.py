# from typing import Optionalfrom typing import List, Optional

from peft import LoraConfig
from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class WhisperTrainingParams(AutoTrainParams):
    """Parameters for Whisper ASR training.
    
    This class defines all parameters needed for fine-tuning Whisper models for Automatic Speech Recognition (ASR),
    including audio processing settings, model configuration, training hyperparameters, and PEFT/LoRA settings.
    
    Attributes:
        sampling_rate (int): Audio sampling rate in Hz. Default is 16000.
        max_duration_secs (float): Maximum duration of audio clips in seconds. Longer clips will be truncated. Default is 30.0.
        preprocessing_num_workers (int): Number of worker processes for audio preprocessing. Default is 4.
        
        model_name (str): Name or path of the Whisper model to fine-tune. Default is "openai/whisper-small".
        language (str): Language code for ASR (e.g., "en" for English). Default is "en".
        task (str): ASR task type ("transcribe" or "translate"). Default is "transcribe".
        
        learning_rate (float): Learning rate for training. Default is 5e-5.
        num_train_epochs (int): Number of training epochs. Default is 3.
        per_device_train_batch_size (int): Training batch size per device. Default is 8.
        per_device_eval_batch_size (int): Evaluation batch size per device. Default is 8.
        gradient_accumulation_steps (int): Number of steps for gradient accumulation. Default is 1.
        eval_accumulation_steps (Optional[int]): Number of steps for gradient accumulation during evaluation. Default is None.
        eval_steps (int): Number of steps between evaluations. Default is 100.
        save_steps (int): Number of steps between model checkpoints. Default is 500.
        logging_steps (int): Number of steps between logging updates. Default is 10.
        max_steps (Optional[int]): Maximum number of training steps. If None, train for num_train_epochs. Default is None.
        warmup_steps (int): Number of warmup steps for learning rate scheduler. Default is 0.
        mixed_precision (str): Mixed precision training type. Default is "fp16".
        log (str): Logging type. Default is "tensorboard".
        
        use_peft (bool): Whether to use PEFT/LoRA for efficient fine-tuning. Default is True.
        lora_r (int): Rank of LoRA matrices. Default is 8.
        lora_alpha (int): LoRA alpha parameter for scaling updates. Default is 32.
        lora_dropout (float): Dropout probability for LoRA layers. Default is 0.1.
        target_modules (List[str]): List of model modules to apply LoRA to. Defaults to attention and feed-forward layers.

        optimizer_type (str): Type of optimizer to use ("adam", "adamw"). Default is "adamw".
        optimizer_beta1 (float): Beta1 parameter for Adam/AdamW optimizers. Default is 0.9.
        optimizer_beta2 (float): Beta2 parameter for Adam/AdamW optimizers. Default is 0.999.
        optimizer_epsilon (float): Epsilon parameter for Adam/AdamW optimizers. Default is 1e-8.
        weight_decay (float): Weight decay for optimizers. Default is 0.0.
        
        lr_scheduler_type (str): Type of learning rate scheduler ("linear", "cosine", "constant", "constant_with_warmup").
                                Default is "linear".
        lr_scheduler_warmup_ratio (float): Ratio of warmup steps relative to total steps. Default is 0.0.
        seed (int): Random seed for reproducibility. Default is 42.
    """
    
    # Data parameters
    data_path: str = Field(None, title="Path to the dataset directory")
    train_split: str = Field("train", title="Name of the training split")
    valid_split: Optional[str] = Field(None, title="Name of the validation split")
    audio_column: str = Field("audio", title="Name of the audio column in the dataset")
    text_column: str = Field("text", title="Name of the text column in the dataset")
    
    # Model parameters
    model: str = Field("openai/whisper-small", title="Name or path of the Whisper model to fine-tune")
    model_name: str = Field("openai/whisper-small", title="Name or path of the Whisper model to fine-tune")
    language: str = Field("en", title="Language code for ASR (e.g., 'en' for English)")
    task: str = Field("transcribe", title="ASR task type ('transcribe' or 'translate')")
    
    # Audio processing parameters
    sampling_rate: int = Field(16000, title="Audio sampling rate in Hz")
    max_duration_secs: float = Field(30.0, title="Maximum duration of audio clips in seconds")
    preprocessing_num_workers: int = Field(4, title="Number of worker processes for audio preprocessing")
    
    # Training hyperparameters
    learning_rate: float = Field(5e-5, title="Learning rate for training")
    num_train_epochs: int = Field(3, title="Number of training epochs")
    per_device_train_batch_size: int = Field(8, title="Training batch size per device")
    per_device_eval_batch_size: int = Field(8, title="Evaluation batch size per device")
    gradient_accumulation_steps: int = Field(1, title="Number of steps for gradient accumulation")
    eval_accumulation_steps: Optional[int] = Field(None, title="Number of steps for gradient accumulation during evaluation")
    eval_steps: int = Field(100, title="Number of steps between evaluations")
    save_steps: int = Field(500, title="Number of steps between model checkpoints")
    logging_steps: int = Field(10, title="Number of steps between logging updates")
    max_steps: Optional[int] = Field(None, title="Maximum number of training steps")
    warmup_steps: int = Field(0, title="Number of warmup steps for learning rate scheduler")
    mixed_precision: str = Field("fp16", title="Mixed precision training type")
    
    # PEFT/LoRA parameters
    use_peft: bool = Field(True, title="Whether to use PEFT/LoRA for efficient fine-tuning")
    lora_r: int = Field(8, title="Rank of LoRA matrices")
    lora_alpha: int = Field(32, title="LoRA alpha parameter for scaling updates")
    lora_dropout: float = Field(0.1, title="Dropout probability for LoRA layers")
    target_modules: List[str] = Field(
        ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        title="List of model modules to apply LoRA to"
    )
    
    # Optimizer parameters
    optimizer_type: str = Field("adamw", title="Type of optimizer to use")
    optimizer_beta1: float = Field(0.9, title="Beta1 parameter for Adam/AdamW optimizers")
    optimizer_beta2: float = Field(0.999, title="Beta2 parameter for Adam/AdamW optimizers")
    optimizer_epsilon: float = Field(1e-8, title="Epsilon parameter for Adam/AdamW optimizers")
    weight_decay: float = Field(0.0, title="Weight decay for optimizers")
    
    # Learning rate scheduler parameters
    lr_scheduler_type: str = Field("linear", title="Type of learning rate scheduler")
    lr_scheduler_warmup_ratio: float = Field(0.0, title="Ratio of warmup steps relative to total steps")
    
    # Other parameters
    seed: int = Field(42, title="Random seed for reproducibility")
    project_name: str = Field("whisper-finetuned", title="Name of the project")
    log: str = Field("tensorboard", title="Logging type")
    
    # Hub parameters
    token: Optional[str] = Field(None, title="Hugging Face token for uploading models")
    push_to_hub: bool = Field(False, title="Whether to push the model to Hugging Face Hub")
    username: Optional[str] = Field(None, title="Hugging Face username")
    
    def get_lora_config(self):
        """
        Returns a LoraConfig object based on the parameters.
        """
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def get_optimizer_kwargs(self) -> dict:
        """Returns optimizer keyword arguments based on the configuration.
        
        Returns:
            dict: Keyword arguments for the optimizer.
        """
        return {
            "beta1": self.optimizer_beta1, 
            "beta2": self.optimizer_beta2,
            "epsilon": self.optimizer_epsilon,
            "weight_decay": self.weight_decay,
        }
    
    def calculate_warmup_steps(self, total_steps: int) -> int:
        """Calculate the number of warmup steps based on ratio or absolute value.
        
        Args:
            total_steps (int): Total number of training steps.
            
        Returns:
            int: Number of warmup steps.
        """
        if self.lr_scheduler_warmup_ratio > 0:
            return int(total_steps * self.lr_scheduler_warmup_ratio)
        return self.warmup_steps 
# from pydantic import Field
# from autotrain.trainers.common import AutoTrainParams


# class ASRParams(AutoTrainParams):
#     """
#     ASRParams is a configuration class for Automatic Speech Recognition (ASR) training parameters.

#     Attributes:
#         data_path (str): Path to the dataset.
#         model (str): Name of the pre-trained model to use. Default is "facebook/wav2vec2-base-960h".
#         lr (float): Learning rate for the optimizer. Default is 3e-4.
#         epochs (int): Number of training epochs. Default is 3.
#         batch_size (int): Batch size for training. Default is 8.
#         warmup_steps (int): Number of warmup steps for learning rate scheduler. Default is 500.
#         gradient_accumulation (int): Number of steps to accumulate gradients before updating. Default is 2.
#         save_steps (int): Number of steps between saving checkpoints. Default is 500.
#         eval_steps (int): Number of steps between evaluations. Default is 500.
#         project_name: str = Field("project-name", title="Name of the project for output directory")
#         train_split: str = Field("train", title="Name of the training data split")
#         valid_split: Optional[str] = Field(None, title="Name of the validation data split")
#         username: Optional[str] = Field(None, title="Hugging Face username for authentication")
#         logging_steps (int): Number of steps between logging. Default is 50.
#         audio_column (str): Name of the column containing audio data. Default is "audio".
#         text_column (str): Name of the column containing text data. Default is "text".
#         lr: float = Field(5e-5, title="Learning rate for the optimizer")
#         warmup_ratio: float = Field(0.1, title="Warmup ratio for learning rate scheduler")
#         optimizer: str = Field("adamw_torch", title="Optimizer type")
#         scheduler: str = Field("linear", title="Learning rate scheduler type")
#         weight_decay: float = Field(0.0, title="Weight decay for the optimizer")
#         max_grad_norm: float = Field(1.0, title="Maximum gradient norm for clipping")
#         seed: int = Field(42, title="Random seed for reproducibility")
#         auto_find_batch_size: bool = Field(False, title="Automatically find optimal batch size")
#         mixed_precision: Optional[str] = Field(None, title="Mixed precision training mode (fp16, bf16, or None)")
#         save_total_limit: int = Field(1, title="Maximum number of checkpoints to keep")
#         token: Optional[str] = Field(None, title="Hugging Face Hub token for authentication")
#         push_to_hub: bool = Field(False, title="Whether to push the model to Hugging Face Hub")
#         eval_strategy: str = Field("epoch", title="Evaluation strategy during training")
#         image_column: str = Field("image", title="Column name for images in the dataset")
#         target_column: str = Field("target", title="Column name for target labels in the dataset")
#         log: str = Field("none", title="Logging method for experiment tracking")
#         early_stopping_patience: int = Field(5, title="Number of epochs with no improvement for early stopping")
#         early_stopping_threshold: float = Field(0.01, title="Threshold for early stopping")
#     """

#     data_path: str = Field(None, title="Data path")
#     model: str = Field("facebook/wav2vec2-base-960h", title="Model name")
#     lr: float = Field(3e-4, title="Learning rate")
#     epochs: int = Field(10, title="Number of training epochs")
#     batch_size: int = Field(16, title="Batch size for training")
#     warmup_steps: int = Field(500, title="Warmup steps")
#     project_name: str = Field("project-name", title="Name of the project for output directory")
#     gradient_accumulation: int = Field(2, title="Gradient accumulation steps")
#     save_steps: int = Field(500, title="Save steps")
#     eval_steps: int = Field(500, title="Evaluation steps")
#     logging_steps: int = Field(50, title="Logging steps")
#     train_split: str = Field("train", title="Name of the training data split")
#     valid_split: Optional[str] = Field(None, title="Name of the validation data split")
#     username: Optional[str] = Field(None, title="Hugging Face username for authentication")
#     token: Optional[str] = Field(None, title="Authentication token for Hugging Face Hub")
#     audio_column: str = Field("audio", title="Audio column")
#     log: str = Field("none", title="Logging method for experiment tracking")
#     text_column: str = Field("text", title="Text column")
#     eval_strategy: str = Field("steps", title="Evaluation strategy")
#     save_total_limit: int = Field(1, title="Save total limit")
#     push_to_hub: bool = Field(False, title="Push to hub")
#     lr: float = Field(5e-5, title="Learning rate for the optimizer")
#     epochs: int = Field(3, title="Number of epochs for training")
#     batch_size: int = Field(8, title="Batch size for training")
#     warmup_ratio: float = Field(0.1, title="Warmup ratio for learning rate scheduler")
#     gradient_accumulation: int = Field(1, title="Number of gradient accumulation steps")
#     optimizer: str = Field("adamw_torch", title="Optimizer type")
#     scheduler: str = Field("linear", title="Learning rate scheduler type")
#     weight_decay: float = Field(0.0, title="Weight decay for the optimizer")
#     max_grad_norm: float = Field(1.0, title="Maximum gradient norm for clipping")
#     seed: int = Field(42, title="Random seed for reproducibility")
#     train_split: str = Field("train", title="Name of the training data split")
#     valid_split: Optional[str] = Field(None, title="Name of the validation data split")
#     logging_steps: int = Field(-1, title="Number of steps between logging")
#     project_name: str = Field("project-name", title="Name of the project for output directory")
#     auto_find_batch_size: bool = Field(False, title="Automatically find optimal batch size")
#     mixed_precision: Optional[str] = Field(None, title="Mixed precision training mode (fp16, bf16, or None)")
#     save_total_limit: int = Field(1, title="Maximum number of checkpoints to keep")
#     token: Optional[str] = Field(None, title="Hugging Face Hub token for authentication")
#     push_to_hub: bool = Field(False, title="Whether to push the model to Hugging Face Hub")
#     eval_strategy: str = Field("epoch", title="Evaluation strategy during training")
#     image_column: str = Field("image", title="Column name for images in the dataset")
#     target_column: str = Field("target", title="Column name for target labels in the dataset")
#     log: str = Field("none", title="Logging method for experiment tracking")
#     early_stopping_patience: int = Field(5, title="Number of epochs with no improvement for early stopping")
#     early_stopping_threshold: float = Field(0.01, title="Threshold for early stopping")