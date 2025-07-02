import argparse
import json
import os
import logging
from typing import Optional, Dict, Any
import torch
import librosa
from datetime import datetime
import pandas as pd
import traceback
import sys

from accelerate.state import PartialState
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModel,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

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
    DetailedTrainingCallback,
)
from autotrain.trainers.automatic_speech_recognition.dataset import AutomaticSpeechRecognitionDataset
from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams
from autotrain.trainers.automatic_speech_recognition.utils import compute_metrics
from autotrain.logger import get_training_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()

def dynamic_padding_collator(batch):
    """Custom collator that handles dynamic padding for different audio lengths."""
    
    input_features = [item['input_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    target_length = 3000  
    
    
    padded_features = []
    for feat in input_features:
        if feat.shape[1] < target_length:
            
            padding = torch.zeros(80, target_length - feat.shape[1])
            padded_feat = torch.cat([feat, padding], dim=1)
        elif feat.shape[1] > target_length:
            
            padded_feat = feat[:, :target_length]
        else:
            padded_feat = feat
        padded_features.append(padded_feat)
    
    
    input_features = torch.stack(padded_features)
    
    
    max_label_length = max(len(label) for label in labels)
    padded_labels = []
    for label in labels:
        if len(label) < max_label_length:
            padding = torch.full((max_label_length - len(label),), -100)  # -100 is ignore_index
            padded_label = torch.cat([label, padding])
        else:
            padded_label = label
        padded_labels.append(padded_label)
    
    labels = torch.stack(padded_labels)
    #hello
    return {
        'input_features': input_features,
        'labels': labels,
    }

def load_data(params, is_validation=False):
    """
    Load dataset from local directory or HuggingFace Hub.

    Args:
        params: Training parameters
        is_validation: Whether to load validation data
        
    Returns:
        Dataset: Loaded dataset
    """
    try:
        if params.using_hub_dataset:
            
            dataset = load_dataset(
                params.data_path,
                split=params.valid_split if is_validation else params.train_split,
                use_auth_token=params.token if params.token else None
            )
        else:
            
            logger.info(f"Looking for data in: {params.data_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            
            if not os.path.exists(params.data_path):
                logger.error(f"Data path does not exist: {params.data_path}")
                raise ValueError(f"Data path does not exist: {params.data_path}")
            
            
            files = os.listdir(params.data_path)
            logger.info(f"Files in directory: {files}")
            
            
            if 'train' in files and 'validation' in files:
                logger.info("Detected HuggingFace dataset format")
                if is_validation:
                    dataset = load_from_disk(os.path.join(params.data_path, 'validation'))
                else:
                    dataset = load_from_disk(os.path.join(params.data_path, 'train'))
                logger.info(f"Loaded {'validation' if is_validation else 'train'} dataset with {len(dataset)} examples")
                return dataset
            
           
            csv_files = [f for f in files if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV file found in {params.data_path}")
            csv_file_path = os.path.join(params.data_path, csv_files[0])
            logger.info(f"Using CSV file: {csv_file_path}")
            
            
            logger.info(f"Loading CSV file: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            
            
            logger.info(f"CSV columns: {df.columns.tolist()}")
            logger.info(f"Number of examples: {len(df)}")
            
           
            if 'audio' in df.columns:
                logger.info("Mapping audio to audio")
                df = df.rename(columns={'audio': 'audio'})
            if 'transcription' in df.columns:
                logger.info("Mapping transcription to transcription")
                df = df.rename(columns={'transcription': 'transcription'})
            
            
            required_columns = ['audio', 'transcription']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}. Available columns: {df.columns.tolist()}")
            
            
            if is_validation:
                
                valid_size = int(len(df) * 0.2)
                df = df.sample(n=valid_size, random_state=42)
                logger.info(f"Using {valid_size} examples for validation")
            else:
                
                train_size = int(len(df) * 0.8)
                df = df.sample(n=train_size, random_state=42)
                logger.info(f"Using {train_size} examples for training")
            
            
            invalid_audio_files = []
            audio_folder = os.path.join(params.data_path, 'audio')
            
            
            if not os.path.exists(audio_folder):
                logger.warning(f"Audio folder not found: {audio_folder}")
                logger.warning("Will try to use audio paths as provided in CSV")
            
            for idx, audio_path in enumerate(df['audio']):
                
                if not os.path.isabs(audio_path):
                    
                    if os.path.exists(audio_folder):
                        audio_file = os.path.join(audio_folder, os.path.basename(audio_path))
                        if os.path.exists(audio_file):
                            df.at[idx, 'audio'] = os.path.abspath(audio_file)
                            continue
                    
                    audio_file = os.path.join(params.data_path, audio_path)
                    if os.path.exists(audio_file):
                        df.at[idx, 'audio'] = os.path.abspath(audio_file)
                        continue
                
               
                if not os.path.exists(audio_path):
                    invalid_audio_files.append((idx, audio_path))
            
            if invalid_audio_files:
                error_msg = "The following audio files were not found:\n"
                for idx, path in invalid_audio_files[:5]:  
                    error_msg += f"Row {idx}: {path}\n"
                if len(invalid_audio_files) > 5:
                    error_msg += f"... and {len(invalid_audio_files) - 5} more files"
                raise ValueError(error_msg)
            
            
            dataset = Dataset.from_pandas(df)
            
            
            if len(dataset) > 0:
                logger.info("First example in dataset:")
                logger.info(f"Audio path: {dataset[0]['audio']}")
                logger.info(f"Transcription: {dataset[0]['transcription']}")
                if 'duration' in dataset[0]:
                    logger.info(f"Duration: {dataset[0]['duration']}")
                
                
                try:
                    audio, sr = librosa.load(dataset[0]['audio'], sr=params.sampling_rate)
                    logger.info(f"Successfully loaded first audio file. Duration: {len(audio)/sr:.2f}s")
                except Exception as e:
                    logger.error(f"Error loading first audio file: {str(e)}")
                    raise
            
            return dataset
            
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def load_model_and_processor(params):
    """
    Load model and processor for ASR training.
    
    Args:
        params: Training parameters
        
    Returns:
        tuple: (model, processor)
    """
    try:
        
        logger.info("Starting processor load for: %s", params.model)
        try:
            processor = AutoProcessor.from_pretrained(
                params.model,
                token=params.token if params.token else None,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
        except Exception as e:
            logger.warning(f"Could not load processor with AutoProcessor, trying Wav2Vec2Processor...")
            try:
                from transformers import Wav2Vec2Processor
                processor = Wav2Vec2Processor.from_pretrained(
                    params.model,
                    token=params.token if params.token else None,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
            except Exception as e2:
                logger.warning(f"Could not load processor with Wav2Vec2Processor, trying WhisperProcessor...")
                try:
                    from transformers import WhisperProcessor
                    processor = WhisperProcessor.from_pretrained(
                        params.model,
                        token=params.token if params.token else None,
                        trust_remote_code=ALLOW_REMOTE_CODE,
                    )
                except Exception as e3:
                    raise ValueError(f"Could not load processor with any known processor type: {str(e3)}")
        
        
        model_name = params.model.lower()
        
        
        try:
            logger.info("Attempting to load as Seq2Seq model...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                params.model,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
            logger.info("Successfully loaded as Seq2Seq model")
        except Exception as e:
            logger.info(f"Could not load as Seq2Seq model: {str(e)}")
            
            try:
                logger.info("Attempting to load as CTC model...")
                model = AutoModelForCTC.from_pretrained(
                    params.model,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
                logger.info("Successfully loaded as CTC model")
            except Exception as e:
                logger.info(f"Could not load as CTC model: {str(e)}")
                
                try:
                    logger.info("Attempting to load as generic model...")
                    model = AutoModel.from_pretrained(
                        params.model,
                        trust_remote_code=ALLOW_REMOTE_CODE,
                    )
                    logger.info("Successfully loaded as generic model")
                except Exception as e:
                    raise ValueError(f"Could not load model {params.model} as any supported type: {str(e)}")
        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Moving model to device: %s", device)
        model = model.to(device)
        logger.info("Model moved to device: %s", device)
        
        
        logger.info(f"Model type: {model.__class__.__name__}")
        logger.info(f"Model configuration: {model.config}")
        
       
        required_methods = ['forward', 'get_encoder', 'get_decoder']
        missing_methods = [method for method in required_methods if not hasattr(model, method)]
        if missing_methods:
            logger.warning(f"Model is missing some required methods: {missing_methods}")
            logger.warning("This might affect training. Please check model documentation.")
        
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading model and processor: {str(e)}")
        raise

@monitor
def train(config: Dict[str, Any]):
    try:
        if isinstance(config, dict):
            config = AutomaticSpeechRecognitionParams(**config)
        training_logger = get_training_logger(getattr(config, 'output_dir', '.'))
        training_logger.info("[LIVE] Initializing ASR training pipeline...")
        training_logger.info("[LIVE] Parameters parsed.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        training_logger.info("[LIVE] Using device: %s", device)

        training_logger.info("[LIVE] Using data_path: %s", config.data_path)
        training_logger.info("[LIVE] Using audio_column: %s", getattr(config, 'audio_column', None))
        training_logger.info("[LIVE] Using text_column: %s", getattr(config, 'text_column', None))

        train_path = os.path.join(config.data_path, "train")
        if os.path.exists(train_path):
            training_logger.info("[LIVE] Loading training dataset from disk...")
            from datasets import load_from_disk
            dataset = load_from_disk(train_path)
            training_logger.info("[LIVE] Training dataset loaded with %d examples.", len(dataset))
        else:
            training_logger.info("[LIVE] Loading dataset using load_data()...")
            dataset = load_data(config)
            training_logger.info("[LIVE] Dataset loaded with %d examples.", len(dataset))
        validation_path = os.path.join(config.data_path, "validation")
        if os.path.exists(validation_path):
            training_logger.info("[LIVE] Loading validation dataset from disk...")
            from datasets import load_from_disk
            valid_dataset = load_from_disk(validation_path)
            training_logger.info("[LIVE] Validation dataset loaded with %d examples.", len(valid_dataset))
        else:
            valid_dataset = None
        training_logger.info("[LIVE] Loading model and processor...")
        model, processor = load_model_and_processor(config)
        from autotrain.trainers.automatic_speech_recognition import utils
        utils.set_processor(processor)
        training_logger.info("[LIVE] Model and processor loaded.")
        training_logger.info("[LIVE] Creating training dataset object...")
        train_dataset = AutomaticSpeechRecognitionDataset(
            data=dataset,
            processor=processor,
            model=model,
            audio_column=config.audio_column,
            text_column=config.text_column,
            max_duration=config.max_duration,
            sampling_rate=config.sampling_rate,
        )
        training_logger.info("[LIVE] Training dataset object created with %d examples.", len(train_dataset))
        if valid_dataset is not None:
            training_logger.info("[LIVE] Creating validation dataset object...")
            valid_dataset_obj = AutomaticSpeechRecognitionDataset(
                data=valid_dataset,
                processor=processor,
                model=model,
                audio_column=config.audio_column,
                text_column=config.text_column,
                max_duration=config.max_duration,
                sampling_rate=config.sampling_rate,
            )
            training_logger.info("[LIVE] Validation dataset object created with %d examples.", len(valid_dataset_obj))
        training_logger.info("[LIVE] Initializing Trainer...")
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=config.lr,
            num_train_epochs=config.epochs,
            save_strategy="epoch",
            disable_tqdm=True,
            evaluation_strategy="epoch" if valid_dataset else "no",
            load_best_model_at_end=True if valid_dataset else False,
            metric_for_best_model="wer" if valid_dataset else None,
            greater_is_better=False if valid_dataset else None,
            push_to_hub=config.push_to_hub,
            hub_model_id=config.hub_model_id,
            hub_token=config.token,
            logging_dir=os.path.join(config.output_dir, "logs"),
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            fp16=config.mixed_precision == "fp16",
            bf16=config.mixed_precision == "bf16",
            dataloader_num_workers=0,  
            dataloader_pin_memory=False,  
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            report_to="tensorboard",
            seed=42,
        )
        training_logger.info("[LIVE] Trainer arguments set. Initializing Trainer...")
        callbacks = [
            LossLoggingCallback(),
            TrainStartCallback(),
            PrinterCallback(),
            DetailedTrainingCallback(),
            EarlyStoppingCallback(early_stopping_patience=3) if valid_dataset is not None else None,
            UploadLogs(config) if config.push_to_hub else None,
        ]
        callbacks = [cb for cb in callbacks if cb is not None]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset_obj if valid_dataset is not None else None,
            callbacks=callbacks,
            data_collator=dynamic_padding_collator,  
            compute_metrics=compute_metrics if valid_dataset is not None else None,  
        )
        training_logger.info("[LIVE] Trainer initialized. Starting training...")
        trainer.train()
        training_logger.info("[LIVE] Training complete.")
        # Save final model and processor to project_name (like image classification)
        trainer.save_model(config.project_name)
        processor.save_pretrained(config.project_name)
        # Create and save model card (FIXED)
        model_card = utils.create_model_card(config, trainer)
        with open(f"{config.project_name}/README.md", "w") as f:
            f.write(model_card)
        # Push model to Hugging Face Hub if push_to_hub is True (main process only)
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
                    folder_path=config.project_name, repo_id=f"{config.username}/{config.project_name}", repo_type="model"
                )
        if PartialState().process_index == 0:
            pause_space(config)
    except Exception as e:
        # Robust error logging if training_logger is not defined
        try:
            training_logger.error("[LIVE] Error in training pipeline: %s", str(e))
        except Exception:
            print(f"[LIVE] Error in training pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, default=None)
    args = parser.parse_args()
    if args.training_config:
        with open(args.training_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        data_path = config.get("data_path", "")
        if data_path == "life_app_data" or os.path.basename(data_path) == "life_app_data":
            from datasets import load_from_disk
            train_dataset = load_from_disk(data_path)
            valid_dataset = None
        else:
            train_dataset = None
            valid_dataset = None
        print(f"Loaded LiFE App dataset: {train_dataset}")
        sys.exit(0)

if __name__ == "__main__":
    args = parse_args()
    with open(args.training_config, "r") as f:
        config = json.load(f)
    train(config)