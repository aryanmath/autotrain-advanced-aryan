import logging
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer

logger = logging.getLogger("autotrain.asr")

RESERVED_COLUMNS = ["autotrain_audio", "autotrain_text"]

@dataclass
class AudioSpeechRecognitionPreprocessor:
    """
    A preprocessor class for Automatic Speech Recognition (ASR) tasks.

    Attributes:
        train_data (str or pd.DataFrame): Path to training data or DataFrame.
        audio_column (str): Name of the column containing audio data.
        text_column (str): Name of the column containing transcription text.
        username (str): Username for Hugging Face Hub.
        project_name (str): Project name for dataset storage.
        token (str): Authentication token for Hugging Face Hub.
        valid_data (Optional[str or pd.DataFrame]): Validation data path or DataFrame. Default is None.
        test_size (Optional[float]): Proportion of data for validation split. Default is 0.2.
        seed (Optional[int]): Random seed for splitting. Default is 42.
        local (Optional[bool]): Save locally or push to Hub. Default is False.
    """
    train_data: str or pd.DataFrame
    audio_column: str
    text_column: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str or pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    def __post_init__(self):
        # Load dataset if train_data is a string (path or Hub ID)
        if isinstance(self.train_data, str):
            self.train_data = load_dataset(self.train_data)["train"].to_pandas()
        if isinstance(self.valid_data, str):
            self.valid_data = load_dataset(self.valid_data)["train"].to_pandas()

        # Validate columns in train_data
        if self.audio_column not in self.train_data.columns:
            raise ValueError(f"{self.audio_column} not in train data")
        if self.text_column not in self.train_data.columns:
            raise ValueError(f"{self.text_column} not in train data")

        # Validate columns in valid_data if provided
        if self.valid_data is not None:
            if self.audio_column not in self.valid_data.columns:
                raise ValueError(f"{self.audio_column} not in valid data")
            if self.text_column not in self.valid_data.columns:
                raise ValueError(f"{self.text_column} not in valid data")

        # Check for reserved columns
        for column in RESERVED_COLUMNS:
            if column in self.train_data.columns:
                raise ValueError(f"{column} is a reserved column name")
            if self.valid_data is not None and column in self.valid_data.columns:
                raise ValueError(f"{column} is a reserved column name")

    def split(self):
        if self.valid_data is not None:
            return self.train_data, self.valid_data
        else:
            train_df, valid_df = train_test_split(
                self.train_data,
                test_size=self.test_size,
                random_state=self.seed,
            )
            train_df = train_df.reset_index(drop=True)
            valid_df = valid_df.reset_index(drop=True)
            return train_df, valid_df

    def prepare_columns(self, train_df, valid_df):
        train_df.loc[:, "autotrain_audio"] = train_df[self.audio_column]
        train_df.loc[:, "autotrain_text"] = train_df[self.text_column]
        valid_df.loc[:, "autotrain_audio"] = valid_df[self.audio_column]
        valid_df.loc[:, "autotrain_text"] = valid_df[self.text_column]

        drop_cols = [self.audio_column, self.text_column]
        train_df = train_df.drop(columns=drop_cols)
        valid_df = valid_df.drop(columns=drop_cols)
        return train_df, valid_df

    def prepare(self):
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)

        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df)

        dataset = DatasetDict({"train": train_dataset, "validation": valid_dataset})

        if self.local:
            dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            return f"{self.project_name}/autotrain-data"
        else:
            dataset["train"].push_to_hub(
                f"{self.username}/autotrain-data-{self.project_name}",
                split="train",
                private=True,
                token=self.token,
            )
            dataset["validation"].push_to_hub(
                f"{self.username}/autotrain-data-{self.project_name}",
                split="validation",
                private=True,
                token=self.token,
            )
            return f"{self.username}/autotrain-data-{self.project_name}"

def train(config):
    """
    Train an ASR model using the provided configuration.

    Args:
        config (dict): Configuration with data_path, model, col_mapping, lr, epochs, batch_size, etc.
    """
    logger.info(f"Starting ASR training with config: {config}")

    # Preprocess data
    data_path = config.get("data_path")
    col_mapping = config.get("col_mapping", {})
    username = config.get("username", "autotrain")
    project_name = config.get("project_name", "asr_project")
    token = config.get("token")

    preprocessor = ASRPreprocessor(
        train_data=data_path,
        audio_column=col_mapping.get("audio", "audio"),
        text_column=col_mapping.get("text", "text"),
        username=username,
        project_name=project_name,
        token=token,
        local=config.get("local", False),
    )
    dataset_path = preprocessor.prepare()
    dataset = load_dataset(dataset_path)

    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained(config["model"])
    model = Wav2Vec2ForCTC.from_pretrained(config["model"])
    logger.info(f"Loaded model and processor: {config['model']}")

    # Preprocess audio and text
    def preprocess(batch):
        audio = [sample["array"] for sample in batch["autotrain_audio"]]
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with processor.as_target_processor():
            labels = processor(batch["autotrain_text"], return_tensors="pt", padding=True)
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": labels.input_ids
        }

    encoded_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing ASR data"
    )
    logger.info("Dataset preprocessed successfully")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config.get("lr", 1e-4),
        num_train_epochs=config.get("epochs", 10),
        per_device_train_batch_size=config.get("batch_size", 8),
        per_device_eval_batch_size=config.get("batch_size", 8),
        evaluation_strategy="epoch" if "validation" in encoded_dataset else "no",
        save_strategy="epoch",
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=10,
        load_best_model_at_end="validation" in encoded_dataset,
    )

    # Custom data collator
    def data_collator(features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]
        return {
            "input_values": torch.stack(input_values),
            "labels": torch.stack(labels)
        }

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset.get("validation"),
        data_collator=data_collator
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model(config["output_dir"])
    logger.info(f"Model saved to {config['output_dir']}")

    # Push to Hub if specified
    if config.get("push_to_hub", False):
        trainer.push_to_hub()
        logger.info("Model pushed to Hugging Face Hub")

    return trainer


# import logging
# import os
# from dataclasses import dataclass
# from typing import Optional

# import pandas as pd
# import torch
# from datasets import Dataset, DatasetDict, load_dataset
# from sklearn.model_selection import train_test_split
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer

# logger = logging.getLogger("autotrain.asr")

# RESERVED_COLUMNS = ["autotrain_audio", "autotrain_text"]

# @dataclass
# class ASRPreprocessor:
#     """
#     Preprocessor for Automatic Speech Recognition (ASR) tasks, mirroring preprocessor folder logic.

#     Attributes:
#         train_data (str or pd.DataFrame): Path to training data or DataFrame.
#         audio_column (str): Column with audio data.
#         text_column (str): Column with transcription text.
#         username (str): Hugging Face Hub username.
#         project_name (str): Project name for dataset storage.
#         token (str): Authentication token for Hugging Face Hub.
#         valid_data (Optional[str or pd.DataFrame]): Validation data. Default is None.
#         test_size (Optional[float]): Validation split proportion. Default is 0.2.
#         seed (Optional[int]): Random seed. Default is 42.
#         local (Optional[bool]): Save locally or push to Hub. Default is False.
#     """
#     train_data: str or pd.DataFrame
#     audio_column: str
#     text_column: str
#     username: str
#     project_name: str
#     token: str
#     valid_data: Optional[str or pd.DataFrame] = None
#     test_size: Optional[float] = 0.2
#     seed: Optional[int] = 42
#     local: Optional[bool] = False

#     def __post_init__(self):
#         # Load dataset if train_data is a string (Hub ID or path)
#         if isinstance(self.train_data, str):
#             self.train_data = load_dataset(self.train_data)["train"].to_pandas()
#         if isinstance(self.valid_data, str):
#             self.valid_data = load_dataset(self.valid_data)["train"].to_pandas()

#         # Validate columns
#         if self.audio_column not in self.train_data.columns:
#             raise ValueError(f"{self.audio_column} not in train data")
#         if self.text_column not in self.train_data.columns:
#             raise ValueError(f"{self.text_column} not in train data")

#         if self.valid_data is not None:
#             if self.audio_column not in self.valid_data.columns:
#                 raise ValueError(f"{self.audio_column} not in valid data")
#             if self.text_column not in self.valid_data.columns:
#                 raise ValueError(f"{self.text_column} not in valid data")

#         # Check reserved columns
#         for column in RESERVED_COLUMNS:
#             if column in self.train_data.columns:
#                 raise ValueError(f"{column} is a reserved column name")
#             if self.valid_data is not None and column in self.valid_data.columns:
#                 raise ValueError(f"{column} is a reserved column name")

#     def split(self):
#         if self.valid_data is not None:
#             return self.train_data, self.valid_data
#         train_df, valid_df = train_test_split(
#             self.train_data,
#             test_size=self.test_size,
#             random_state=self.seed,
#         )
#         train_df = train_df.reset_index(drop=True)
#         valid_df = valid_df.reset_index(drop=True)
#         return train_df, valid_df

#     def prepare_columns(self, train_df, valid_df):
#         train_df.loc[:, "autotrain_audio"] = train_df[self.audio_column]
#         train_df.loc[:, "autotrain_text"] = train_df[self.text_column]
#         valid_df.loc[:, "autotrain_audio"] = valid_df[self.audio_column]
#         valid_df.loc[:, "autotrain_text"] = valid_df[self.text_column]

#         drop_cols = [self.audio_column, self.text_column]
#         train_df = train_df.drop(columns=drop_cols)
#         valid_df = valid_df.drop(columns=drop_cols)
#         return train_df, valid_df

#     def prepare(self):
#         train_df, valid_df = self.split()
#         train_df, valid_df = self.prepare_columns(train_df, valid_df)

#         train_dataset = Dataset.from_pandas(train_df)
#         valid_dataset = Dataset.from_pandas(valid_df)

#         dataset = DatasetDict({"train": train_dataset, "validation": valid_dataset})

#         if self.local:
#             dataset.save_to_disk(f"{self.project_name}/autotrain-data")
#             return f"{self.project_name}/autotrain-data"
#         dataset["train"].push_to_hub(
#             f"{self.username}/autotrain-data-{self.project_name}",
#             split="train",
#             private=True,
#             token=self.token,
#         )
#         dataset["validation"].push_to_hub(
#             f"{self.username}/autotrain-data-{self.project_name}",
#             split="validation",
#             private=True,
#             token=self.token,
#         )
#         return f"{self.username}/autotrain-data-{self.project_name}"

# def train(config):
#     """
#     Train an ASR model for task ID 32 ('asr').

#     Args:
#         config (dict): Configuration with data_path, model, col_mapping, lr, epochs, batch_size, etc.
#     """
#     logger.info(f"Starting ASR training (task ID: 32) with config: {config}")

#     # Extract config values
#     data_path = config.get("data_path")
#     col_mapping = config.get("col_mapping", {"audio": "audio", "text": "text"})
#     username = config.get("username", "autotrain")
#     project_name = config.get("project_name", "asr_project")
#     token = config.get("token")
#     local = config.get("local", False)

#     # Preprocess data
#     preprocessor = ASRPreprocessor(
#         train_data=data_path,
#         audio_column=col_mapping.get("audio", "audio"),
#         text_column=col_mapping.get("text", "text"),
#         username=username,
#         project_name=project_name,
#         token=token,
#         local=local,
#     )
#     dataset_path = preprocessor.prepare()
#     dataset = load_dataset(dataset_path)

#     # Load processor and model
#     processor = Wav2Vec2Processor.from_pretrained(config["model"])
#     model = Wav2Vec2ForCTC.from_pretrained(config["model"])
#     logger.info(f"Loaded model and processor: {config['model']}")

#     # Preprocess audio and text
#     def preprocess(batch):
#         audio = [sample["array"] for sample in batch["autotrain_audio"]]
#         inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
#         with processor.as_target_processor():
#             labels = processor(batch["autotrain_text"], return_tensors="pt", padding=True)
#         return {
#             "input_values": inputs.input_values.squeeze(0),
#             "labels": labels.input_ids
#         }

#     encoded_dataset = dataset.map(
#         preprocess,
#         batched=True,
#         remove_columns=dataset["train"].column_names,
#         desc="Preprocessing ASR data"
#     )
#     logger.info("Dataset preprocessed successfully")

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir=config["output_dir"],
#         learning_rate=config.get("lr", 1e-4),
#         num_train_epochs=config.get("epochs", 10),
#         per_device_train_batch_size=config.get("batch_size", 8),
#         per_device_eval_batch_size=config.get("batch_size", 8),
#         evaluation_strategy="epoch" if "validation" in encoded_dataset else "no",
#         save_strategy="epoch",
#         logging_dir=f"{config['output_dir']}/logs",
#         logging_steps=10,
#         load_best_model_at_end="validation" in encoded_dataset,
#     )

#     # Data collator
#     def data_collator(features):
#         input_values = torch.stack([torch.tensor(f["input_values"]) for f in features])
#         labels = torch.stack([torch.tensor(f["labels"]) for f in features])
#         return {"input_values": input_values, "labels": labels}

#     # Initialize trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=encoded_dataset["train"],
#         eval_dataset=encoded_dataset.get("validation"),
#         data_collator=data_collator
#     )

#     # Train
#     logger.info("Starting training...")
#     trainer.train()

#     # Save model
#     trainer.save_model(config["output_dir"])
#     logger.info(f"Model saved to {config['output_dir']}")

#     # Push to Hub if specified
#     if config.get("push_to_hub", False):
#         trainer.push_to_hub()
#         logger.info("Model pushed to Hugging Face Hub")

#     return trainer

# from transformers import Wav2Vec2Processor
# import torchaudio               


# class AudioSpeechRecognitionPreprocessor:
#     """
#     A preprocessor class for audio speech recognition tasks.

#     Args:
#         processor (Wav2Vec2Processor): The processor to preprocess the audio data.

#     Attributes:
#         processor (Wav2Vec2Processor): The processor to preprocess the audio data.

#     Methods:
#         process(audio_path): Processes the audio file and returns input values and attention mask.
#     """

#     def __init__(self, processor):
#         self.processor = processor

#     def process(self, audio_path):
#         """Processes the audio file and returns input values and attention mask."""
#         waveform, sample_rate = torchaudio.load(audio_path)
#         inputs = self.processor(
#             waveform.squeeze(),
#             sampling_rate=sample_rate,
#             return_tensors="pt",
#             padding="longest",
#         )
#         return inputs.input_values.squeeze(), inputs.attention_mask.squeeze()
