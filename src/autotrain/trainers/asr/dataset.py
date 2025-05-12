import torch
from datasets import Dataset
from transformers import Wav2Vec2Processor


class ASRDataset:
    """
    ASRDataset is a class for preprocessing ASR datasets.

    Attributes:
        data (Dataset): The dataset to preprocess.
        processor (Wav2Vec2Processor): The processor for tokenizing audio and text.
        config (ASRParams): The configuration parameters for the ASR task.
    """

    def __init__(self, data: Dataset, processor: Wav2Vec2Processor, config):
        self.data = data
        self.processor = processor
        self.config = config
        self.data = self.data.map(self._prepare_dataset, remove_columns=self.data.column_names)

    def _prepare_dataset(self, batch):
        audio = batch[self.config.audio_column]
        text = batch[self.config.text_column]

        inputs = self.processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        with self.processor.as_target_processor():
            labels = self.processor(text).input_ids

        inputs["labels"] = labels
        return inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]