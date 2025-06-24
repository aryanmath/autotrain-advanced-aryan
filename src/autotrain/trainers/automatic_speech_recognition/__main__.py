import argparse
import json
import os
from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk, Dataset
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
from autotrain.trainers.automatic_speech_recognition.dataset import AutomaticSpeechRecognitionDataset
from autotrain.trainers.automatic_speech_recognition.utils import compute_metrics, create_model_card

CACHE_DIR = r"C:/Users/Aryan/.cache/huggingface/hub/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


def load_data(config):
    # Support local CSV/JSON, local folder, or hub dataset
    if config.using_hub_dataset:
        logger.info("Loading dataset from HuggingFace Hub...")
        train_data = load_dataset(
            config.data_path,
            split=config.train_split,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
        valid_data = load_dataset(
            config.data_path,
            split=config.valid_split,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        ) if config.valid_split else None
    elif os.path.isdir(config.data_path):
        logger.info("Loading dataset from disk...")
        ds = load_from_disk(config.data_path)
        train_data = ds["train"] if "train" in ds else ds[0]
        valid_data = ds["validation"] if "validation" in ds else (ds["test"] if "test" in ds else None)
    elif config.data_path.endswith(".csv"):
        logger.info("Loading dataset from CSV file...")
        ds = Dataset.from_csv(config.data_path)
        train_data, valid_data = None, None
        if "split" in ds.column_names:
            train_data = ds.filter(lambda x: x["split"] == "train")
            valid_data = ds.filter(lambda x: x["split"] == "validation")
        else:
            ds = ds.train_test_split(test_size=0.2, seed=config.seed)
            train_data = ds["train"]
            valid_data = ds["test"]
    elif config.data_path.endswith(".json"):
        logger.info("Loading dataset from JSON file...")
        ds = Dataset.from_json(config.data_path)
        ds = ds.train_test_split(test_size=0.2, seed=config.seed)
        train_data = ds["train"]
        valid_data = ds["test"]
    else:
        raise ValueError(f"Unsupported data format: {config.data_path}")
    return train_data, valid_data


@monitor
def train(config):
    if isinstance(config, dict):
        config = AutomaticSpeechRecognitionParams(**config)
    config.validate_params()

    logger.info("Loading data...")
    train_data, valid_data = load_data(config)

    logger.info("Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            config.model,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise

    logger.info("Preparing datasets...")
    train_dataset = AutomaticSpeechRecognitionDataset(
        data=train_data,
        processor=processor,
        config=config,
        audio_column=config.audio_column,
        text_column=config.text_column,
        max_duration=config.max_duration,
        sampling_rate=config.sampling_rate,
    )
    valid_dataset = None
    if valid_data is not None:
        valid_dataset = AutomaticSpeechRecognitionDataset(
            data=valid_data,
            processor=processor,
            config=config,
            audio_column=config.audio_column,
            text_column=config.text_column,
            max_duration=config.max_duration,
            sampling_rate=config.sampling_rate,
        )

    logger.info("Loading model...")
    try:
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                config.model,
                token=config.token,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
        except Exception:
            model = AutoModelForCTC.from_pretrained(
                config.model,
                token=config.token,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.mixed_precision == "fp16",
        bf16=config.mixed_precision == "bf16",
        evaluation_strategy=config.eval_strategy if valid_dataset is not None else "no",
        save_strategy=config.eval_strategy if valid_dataset is not None else "no",
        load_best_model_at_end=True if valid_dataset is not None else False,
        report_to=config.log,
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        ddp_find_unused_parameters=False,
        push_to_hub=False,
        seed=config.seed,
    )

    logger.info("Setting up callbacks...")
    callbacks = [UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()]
    if valid_dataset is not None:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        ))
    callbacks.append(PrinterCallback())

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
        callbacks=callbacks,
        compute_metrics=compute_metrics if valid_dataset is not None else None,
    )

    logger.info("Starting training...")
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    logger.info("Saving model and processor...")
    trainer.save_model(config.project_name)
    processor.save_pretrained(config.project_name)

    logger.info("Creating model card...")
    model_card = create_model_card(config, trainer)
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub:
        if PartialState().process_index == 0:
            remove_autotrain_data(config)
            save_training_params(config)
            logger.info("Pushing model to hub...")
            api = HfApi(token=config.token)
            api.create_repo(
                repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
            )
            api.upload_folder(
                folder_path=config.project_name,
                repo_id=f"{config.username}/{config.project_name}",
                repo_type="model",
            )
    if PartialState().process_index == 0:
        pause_space(config)


if __name__ == "__main__":
    args = parse_args()
    with open(args.training_config, "r") as f:
        training_config = json.load(f)
    config = AutomaticSpeechRecognitionParams(**training_config)
    train(config)





# import argparse
# import json
# import os
# import logging
# from typing import Optional, Dict, Any
# import torch
# import librosa
# from datetime import datetime
# import pandas as pd
# import traceback
# import sys

# from accelerate.state import PartialState
# from datasets import load_from_disk, load_dataset, Dataset
# from huggingface_hub import HfApi
# from transformers import (
#     AutoConfig,
#     AutoModelForCTC,
#     AutoModelForSpeechSeq2Seq,
#     AutoProcessor,
#     AutoModel,
#     EarlyStoppingCallback,
#     Trainer,
#     TrainingArguments,
# )
# from transformers.trainer_callback import PrinterCallback
# from transformers.trainer_utils import get_last_checkpoint

# from autotrain import logger
# from autotrain.trainers.common import (
#     ALLOW_REMOTE_CODE,
#     LossLoggingCallback,
#     TrainStartCallback,
#     UploadLogs,
#     monitor,
#     pause_space,
#     remove_autotrain_data,
#     save_training_params,
#     DetailedTrainingCallback,
# )
# from autotrain.trainers.automatic_speech_recognition.dataset import AutomaticSpeechRecognitionDataset
# from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
# from autotrain.trainers.automatic_speech_recognition.utils import compute_metrics
# from autotrain.logger import get_training_logger

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--training_config", type=str, required=True)
#     return parser.parse_args()

# def load_data(params, is_validation=False):
#     """
#     Load dataset from local directory or HuggingFace Hub.

#     Args:
#         params: Training parameters
#         is_validation: Whether to load validation data
        
#     Returns:
#         Dataset: Loaded dataset
#     """
#     try:
#         if params.using_hub_dataset:
#             # Load from HuggingFace Hub
#             dataset = load_dataset(
#                 params.data_path,
#                 split=params.valid_split if is_validation else params.train_split,
#                 use_auth_token=params.token if params.token else None
#             )
#         else:
#             # Load from local directory
#             logger.info(f"Looking for data in: {params.data_path}")
#             logger.info(f"Current working directory: {os.getcwd()}")
            
#             # Check if data_path exists
#             if not os.path.exists(params.data_path):
#                 logger.error(f"Data path does not exist: {params.data_path}")
#                 raise ValueError(f"Data path does not exist: {params.data_path}")
            
#             # List all files in the directory
#             files = os.listdir(params.data_path)
#             logger.info(f"Files in directory: {files}")
            
#             # Look for any CSV file
#             csv_files = [f for f in files if f.endswith('.csv')]
#             if not csv_files:
#                 raise ValueError(f"No CSV file found in {params.data_path}")
#             csv_file_path = os.path.join(params.data_path, csv_files[0])
#             logger.info(f"Using CSV file: {csv_file_path}")
            
#             # Read CSV file using pandas
#             logger.info(f"Loading CSV file: {csv_file_path}")
#             df = pd.read_csv(csv_file_path)
            
#             # Log dataset info
#             logger.info(f"CSV columns: {df.columns.tolist()}")
#             logger.info(f"Number of examples: {len(df)}")
            
#             # Map column names if needed
#             if 'autotrain_audio' in df.columns:
#                 logger.info("Mapping autotrain_audio to audio")
#                 df = df.rename(columns={'autotrain_audio': 'audio'})
#             if 'autotrain_transcription' in df.columns:
#                 logger.info("Mapping autotrain_transcription to transcription")
#                 df = df.rename(columns={'autotrain_transcription': 'transcription'})
            
#             # Verify required columns
#             required_columns = ['audio', 'transcription']
#             missing_columns = [col for col in required_columns if col not in df.columns]
#             if missing_columns:
#                 raise ValueError(f"Missing required columns in CSV: {missing_columns}. Available columns: {df.columns.tolist()}")
            
#             # Split into train/valid if needed
#             if is_validation:
#                 # Use 20% of data for validation
#                 valid_size = int(len(df) * 0.2)
#                 df = df.sample(n=valid_size, random_state=42)
#                 logger.info(f"Using {valid_size} examples for validation")
#             else:
#                 # Use 80% of data for training
#                 train_size = int(len(df) * 0.8)
#                 df = df.sample(n=train_size, random_state=42)
#                 logger.info(f"Using {train_size} examples for training")
            
#             # Verify audio files exist and update paths if needed
#             invalid_audio_files = []
#             audio_folder = os.path.join(params.data_path, 'audio')
            
#             # Check if audio folder exists
#             if not os.path.exists(audio_folder):
#                 logger.warning(f"Audio folder not found: {audio_folder}")
#                 logger.warning("Will try to use audio paths as provided in CSV")
            
#             for idx, audio_path in enumerate(df['audio']):
#                 # Convert to absolute path if relative
#                 if not os.path.isabs(audio_path):
#                     # Try to find audio file in audio folder
#                     if os.path.exists(audio_folder):
#                         audio_file = os.path.join(audio_folder, os.path.basename(audio_path))
#                         if os.path.exists(audio_file):
#                             df.at[idx, 'audio'] = os.path.abspath(audio_file)
#                             continue
#                     # If not found in audio folder, try relative to data_path
#                     audio_file = os.path.join(params.data_path, audio_path)
#                     if os.path.exists(audio_file):
#                         df.at[idx, 'audio'] = os.path.abspath(audio_file)
#                         continue
                
#                 # If absolute path, check if it exists
#                 if not os.path.exists(audio_path):
#                     invalid_audio_files.append((idx, audio_path))
            
#             if invalid_audio_files:
#                 error_msg = "The following audio files were not found:\n"
#                 for idx, path in invalid_audio_files[:5]:  # Show first 5 errors
#                     error_msg += f"Row {idx}: {path}\n"
#                 if len(invalid_audio_files) > 5:
#                     error_msg += f"... and {len(invalid_audio_files) - 5} more files"
#                 raise ValueError(error_msg)
            
#             # Convert pandas DataFrame to Datasets Dataset
#             dataset = Dataset.from_pandas(df)
            
#             # Log first example for verification
#             if len(dataset) > 0:
#                 logger.info("First example in dataset:")
#                 logger.info(f"Audio path: {dataset[0]['audio']}")
#                 logger.info(f"Transcription: {dataset[0]['transcription']}")
#                 if 'duration' in dataset[0]:
#                     logger.info(f"Duration: {dataset[0]['duration']}")
                
#                 # Verify audio file can be loaded
#                 try:
#                     audio, sr = librosa.load(dataset[0]['audio'], sr=params.sampling_rate)
#                     logger.info(f"Successfully loaded first audio file. Duration: {len(audio)/sr:.2f}s")
#                 except Exception as e:
#                     logger.error(f"Error loading first audio file: {str(e)}")
#                     raise
            
#             return dataset
            
#     except Exception as e:
#         logger.error(f"Error loading dataset: {str(e)}")
#         raise

# def load_model_and_processor(params):
#     """
#     Load model and processor for ASR training.
    
#     Args:
#         params: Training parameters
        
#     Returns:
#         tuple: (model, processor)
#     """
#     try:
#         # Load processor
#         # logger.info(f"Loading processor: {params.model}")
#         logger.info("Starting processor load for: %s", params.model)
#         try:
#             processor = AutoProcessor.from_pretrained(
#                 params.model,
#                 token=params.token if params.token else None,
#                 trust_remote_code=ALLOW_REMOTE_CODE,
#             )
#         except Exception as e:
#             logger.warning(f"Could not load processor with AutoProcessor, trying Wav2Vec2Processor...")
#             try:
#                 from transformers import Wav2Vec2Processor
#                 processor = Wav2Vec2Processor.from_pretrained(
#                     params.model,
#                     token=params.token if params.token else None,
#                     trust_remote_code=ALLOW_REMOTE_CODE,
#                 )
#             except Exception as e2:
#                 logger.warning(f"Could not load processor with Wav2Vec2Processor, trying WhisperProcessor...")
#                 try:
#                     from transformers import WhisperProcessor
#                     processor = WhisperProcessor.from_pretrained(
#                         params.model,
#                         token=params.token if params.token else None,
#                         trust_remote_code=ALLOW_REMOTE_CODE,
#                     )
#                 except Exception as e3:
#                     raise ValueError(f"Could not load processor with any known processor type: {str(e3)}")
        
#         # Determine model type and load appropriate model
#         model_name = params.model.lower()
        
#         # Try loading as Seq2Seq model first (for Whisper, MMS, etc.)
#         try:
#             logger.info("Attempting to load as Seq2Seq model...")
#             model = AutoModelForSpeechSeq2Seq.from_pretrained(
#                 params.model,
#                 token=params.token if params.token else None,
#                 trust_remote_code=ALLOW_REMOTE_CODE,
#             )
#             logger.info("Successfully loaded as Seq2Seq model")
#         except Exception as e:
#             logger.info(f"Could not load as Seq2Seq model: {str(e)}")
#             # Try loading as CTC model (for Wav2Vec2, Hubert, etc.)
#             try:
#                 logger.info("Attempting to load as CTC model...")
#                 model = AutoModelForCTC.from_pretrained(
#                     params.model,
#                     token=params.token if params.token else None,
#                     trust_remote_code=ALLOW_REMOTE_CODE,
#                 )
#                 logger.info("Successfully loaded as CTC model")
#             except Exception as e:
#                 logger.info(f"Could not load as CTC model: {str(e)}")
#                 # Try loading as generic model
#                 try:
#                     logger.info("Attempting to load as generic model...")
#                     model = AutoModel.from_pretrained(
#                         params.model,
#                         token=params.token if params.token else None,
#                         trust_remote_code=ALLOW_REMOTE_CODE,
#                     )
#                     logger.info("Successfully loaded as generic model")
#                 except Exception as e:
#                     raise ValueError(f"Could not load model {params.model} as any supported type: {str(e)}")
        
#         # Move model to device
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info("Moving model to device: %s", device)
#         model = model.to(device)
#         logger.info("Model moved to device: %s", device)
        
#         # Log model configuration
#         logger.info(f"Model type: {model.__class__.__name__}")
#         logger.info(f"Model configuration: {model.config}")
        
#         # Check if model has required methods
#         required_methods = ['forward', 'get_encoder', 'get_decoder']
#         missing_methods = [method for method in required_methods if not hasattr(model, method)]
#         if missing_methods:
#             logger.warning(f"Model is missing some required methods: {missing_methods}")
#             logger.warning("This might affect training. Please check model documentation.")
        
#         # Check model parameters
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         logger.info(f"Total parameters: {total_params:,}")
#         logger.info(f"Trainable parameters: {trainable_params:,}")
        
#         return model, processor
        
#     except Exception as e:
#         logger.error(f"Error loading model and processor: {str(e)}")
#         raise

# @monitor
# def train(config: Dict[str, Any]):
#     try:
#         # Setup detailed logging
#         training_logger = get_training_logger(config.get('output_dir', '.'))
#         training_logger.info("[LIVE] Initializing ASR training pipeline...")
#         params = AutomaticSpeechRecognitionParams(**config)
#         training_logger.info("[LIVE] Parameters parsed.")
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         training_logger.info("[LIVE] Using device: %s", device)

#         # Log the actual data_path and column mapping being used
#         training_logger.info("[LIVE] Using data_path: %s", params.data_path)
#         training_logger.info("[LIVE] Using audio_column: %s", getattr(params, 'audio_column', None))
#         training_logger.info("[LIVE] Using text_column: %s", getattr(params, 'text_column', None))

#         # Load dataset
#         training_logger.info("[LIVE] Loading dataset...")
#         dataset = load_data(params)
#         training_logger.info("[LIVE] Dataset loaded with %d examples.", len(dataset))
#         if params.valid_split:
#             training_logger.info("[LIVE] Loading validation dataset...")
#             valid_dataset = load_data(params, is_validation=True)
#             training_logger.info("[LIVE] Validation dataset loaded with %d examples.", len(valid_dataset))
#         else:
#             valid_dataset = None
#         training_logger.info("[LIVE] Loading model and processor...")
#         model, processor = load_model_and_processor(params)
#         training_logger.info("[LIVE] Model and processor loaded.")
#         training_logger.info("[LIVE] Creating training dataset object...")
#         train_dataset = AutomaticSpeechRecognitionDataset(
#             data=dataset,
#             processor=processor,
#             config=params,
#             audio_column=params.audio_column,
#             text_column=params.text_column,
#             max_duration=params.max_duration,
#             sampling_rate=params.sampling_rate,
#         )
#         training_logger.info("[LIVE] Training dataset object created with %d examples.", len(train_dataset))
#         if valid_dataset is not None:
#             training_logger.info("[LIVE] Creating validation dataset object...")
#             valid_dataset_obj = AutomaticSpeechRecognitionDataset(
#                 data=valid_dataset,
#                 processor=processor,
#                 config=params,
#                 audio_column=params.audio_column,
#                 text_column=params.text_column,
#                 max_duration=params.max_duration,
#                 sampling_rate=params.sampling_rate,
#             )
#             training_logger.info("[LIVE] Validation dataset object created with %d examples.", len(valid_dataset_obj))
#         training_logger.info("[LIVE] Initializing Trainer...")
#         training_args = TrainingArguments(
#             output_dir=params.output_dir,
#             per_device_train_batch_size=params.batch_size,
#             per_device_eval_batch_size=params.batch_size,
#             gradient_accumulation_steps=params.gradient_accumulation,
#             learning_rate=params.lr,
#             num_train_epochs=params.epochs,
#             save_strategy="epoch",
#             evaluation_strategy="epoch" if valid_dataset else "no",
#             load_best_model_at_end=True if valid_dataset else False,
#             metric_for_best_model="wer" if valid_dataset else None,
#             greater_is_better=False if valid_dataset else None,
#             push_to_hub=params.push_to_hub,
#             hub_model_id=params.hub_model_id,
#             hub_token=params.token,
#             logging_dir=os.path.join(params.output_dir, "logs"),
#             logging_steps=10,
#             save_total_limit=2,
#             remove_unused_columns=False,
#             fp16=params.mixed_precision == "fp16",
#             bf16=params.mixed_precision == "bf16",
#             dataloader_num_workers=4,
#             dataloader_pin_memory=True,
#             gradient_checkpointing=True,
#             optim="adamw_torch",
#             lr_scheduler_type="linear",
#             warmup_ratio=0.1,
#             weight_decay=0.01,
#             max_grad_norm=1.0,
#             report_to="tensorboard",
#             seed=42,
#         )
#         training_logger.info("[LIVE] Trainer arguments set. Initializing Trainer...")
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=valid_dataset_obj if valid_dataset is not None else None,
#             callbacks=[
#                 LossLoggingCallback(),
#                 TrainStartCallback(),
#                 PrinterCallback(),
#                 DetailedTrainingCallback(),
#                 EarlyStoppingCallback(early_stopping_patience=3) if valid_dataset is not None else None,
#                 UploadLogs(params) if params.push_to_hub else None,
#             ],
#         )
#         training_logger.info("[LIVE] Trainer initialized. Starting training...")
#         trainer.train()
#         training_logger.info("[LIVE] Training complete.")
#     except Exception as e:
#         training_logger.error("[LIVE] Error in training pipeline: %s", str(e))
#         logger.error(traceback.format_exc())
#         raise

# def main():
#     # Support for running as a script with --training_config argument
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--training_config", type=str, default=None)
#     args = parser.parse_args()
#     if args.training_config:
#         with open(args.training_config, "r", encoding="utf-8") as f:
#             config = json.load(f)
#         # If data_path is "life_app_data", load from disk
#         data_path = config.get("data_path", "")
#         if data_path == "life_app_data" or os.path.basename(data_path) == "life_app_data":
#             from datasets import load_from_disk
#             train_dataset = load_from_disk(data_path)
#             valid_dataset = None
#         else:
#             # Load from other sources as per existing logic
#             train_dataset = None
#             valid_dataset = None
#         # For demonstration, print loaded dataset info
#         print(f"Loaded LiFE App dataset: {train_dataset}")
#         # Call your training function here, e.g. train_asr(train_dataset, ...)
#         sys.exit(0)
#     try:
#         # Load training config
#         with open("training_config.json", "r") as f:
#             training_config = json.load(f)
            
#         logger.info("Initializing ASR training...")
        
#         # Initialize trainer with detailed logging
#         trainer = automatic_speech_recognitionTrainer(
#             training_config=training_config,
#             callbacks=[DetailedTrainingCallback()]  # Add detailed logging callback
#         )
        
#         # Start training with detailed progress
#         logger.info("Starting ASR training with detailed progress logging...")
#         trainer.train()
        
#     except Exception as e:
#         logger.error(f"Error during ASR training: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise e

# if __name__ == "__main__":
#     args = parse_args()
#     with open(args.training_config, "r") as f:
#         config = json.load(f)
#     train(config)