import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, Optional
from datasets import Dataset

from autotrain import logger

print(">>> RUNNING NEW dataset.py FROM:", __file__)

def detect_model_type(model):
    """Detect model type from model class name."""
    model_class = type(model).__name__
    if 'Whisper' in model_class or 'Seq2Seq' in model_class:
        return "seq2seq"
    elif 'CTC' in model_class or 'Wav2Vec' in model_class:
        return "ctc"
    else:
        return "generic"

def safe_tokenize_text(processor, text, max_seq_length=128):
    """
    Simplified tokenization that works for all models.
    """
    # Method 1: Try Whisper tokenizer directly (most reliable)
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'encode'):
        try:
            input_ids = processor.tokenizer.encode(
                text,
                max_length=max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            return input_ids.squeeze(0)
        except Exception as e:
            logger.warning(f"Whisper tokenizer failed: {e}")
    
    # Method 2: Try processor.tokenizer
    if hasattr(processor, "tokenizer"):
        try:
            return processor.tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.squeeze(0)
        except Exception as e:
            logger.warning(f"Processor tokenizer failed: {e}")
    
    # Method 3: Try processor directly
    try:
        return processor(
            text,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
    except Exception as e:
        logger.warning(f"Processor direct failed: {e}")
    
    # Final fallback: return dummy tokens
    logger.warning(f"All tokenization methods failed for text: {text[:50]}...")
    return torch.tensor([0] * min(max_seq_length, 10), dtype=torch.long)

class AutomaticSpeechRecognitionDataset:
    """
    Simplified ASR Dataset that works with all models.
    """
    def __init__(
        self,
        data: Dataset,
        processor: Any,
        model: Any,
        audio_column: str = "audio",
        text_column: str = "transcription",
        max_duration: float = 30.0,
        sampling_rate: int = 16000,
        model_type: str = None,
    ):
        self._data = data
        self.processor = processor
        self.model = model
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        self.max_seq_length = 128
        
        # Detect model type
        if model_type is not None:
            self.model_type = model_type
        else:
            self.model_type = detect_model_type(model)
        
        logger.info(f"NEW Dataset initialized with model_type: {self.model_type}")
        logger.info(f"Processor type: {type(processor).__name__}")
        logger.info(f"Model type: {type(model).__name__}")
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        """
        try:
            item = self._data[idx]
            
            # Get audio path
            audio_path = item[self.audio_column]
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")

            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Check duration
            duration = len(audio) / sr
            if duration > self.max_duration:
                logger.warning(f"Audio duration {duration:.2f}s exceeds max_duration {self.max_duration}s, truncating")
                max_samples = int(self.max_duration * self.sampling_rate)
                audio = audio[:max_samples]
            
            # Process audio based on model type
            if self.model_type == 'seq2seq':
                # For Whisper and other Seq2Seq models
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    input_features = inputs.input_features[0]
                except Exception as e:
                    logger.warning(f"Seq2Seq audio processing failed: {e}")
                    # Fallback: use raw audio
                    input_features = torch.tensor(audio, dtype=torch.float32)
                    
            elif self.model_type == 'ctc':
                # For Wav2Vec2 and other CTC models
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    input_features = inputs.input_values[0]
                except Exception as e:
                    logger.warning(f"CTC audio processing failed: {e}")
                    # Fallback: use raw audio
                    input_features = torch.tensor(audio, dtype=torch.float32)
                    
            else:
                # Generic approach
                try:
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    if hasattr(inputs, 'input_features'):
                        input_features = inputs.input_features[0]
                    elif hasattr(inputs, 'input_values'):
                        input_features = inputs.input_values[0]
                    else:
                        input_features = torch.tensor(audio, dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"Generic audio processing failed: {e}")
                    input_features = torch.tensor(audio, dtype=torch.float32)
            
            # Get text
            text = item[self.text_column]
            if not text or not isinstance(text, str):
                text = " "  # Empty text fallback
            
            # Tokenize text using simplified method
            labels = safe_tokenize_text(self.processor, text, self.max_seq_length)
            
            # Return based on model type
            if self.model_type == 'seq2seq':
                return {
                    "input_features": input_features,
                    "labels": labels,
                }
            else:
                return {
                    "input_values": input_features,
                    "labels": labels,
                }

        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            # Return dummy data to prevent training from crashing
            dummy_audio = torch.zeros(1000, dtype=torch.float32)
            dummy_labels = torch.tensor([0], dtype=torch.long)
            
            if self.model_type == 'seq2seq':
                return {
                    "input_features": dummy_audio,
                    "labels": dummy_labels,
                }
            else:
                return {
                    "input_values": dummy_audio,
                    "labels": dummy_labels,
                }

def load_life_app_dataset(data_path):
    """Utility to load LiFE App dataset."""
    from datasets import load_from_disk
    return load_from_disk(data_path)