import torch
import torchaudio

class AudioSpeechRecognitionDataset:
    """
    A dataset class for audio speech recognition tasks.

    Args:
        data (list): The dataset containing audio paths and transcription columns.
        processor (Wav2Vec2Processor): The processor to preprocess the audio data.
        config (object): Configuration object containing dataset parameters.

    Attributes:
        data (list): The dataset containing audio paths and transcription columns.
        processor (Wav2Vec2Processor): The processor to preprocess the audio data.
        config (object): Configuration object containing dataset parameters.
        audio_column (str): The name of the column containing audio file paths.
        transcription_column (str): The name of the column containing transcriptions.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(item): Returns a dictionary containing processed audio input and transcriptions for the given item index.
    """

    def __init__(self, data, processor, config):
        self.data = data
        self.processor = processor
        self.config = config
        self.audio_column = self.config.audio_column
        self.text_column = self.config.text_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Retrieve audio path and transcription text
        audio_path = self.data[item][self.audio_column]
        text = self.data[item][self.text_column]

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Process audio into input_values and attention_mask
        inputs = self.processor(
            waveform.squeeze(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        input_values = inputs.input_values.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        # Process transcription text into token IDs
        labels = self.processor.tokenizer(text).input_ids

        # Return dictionary with processed data
        return {
            "input_values": input_values,    # Tensor of processed audio features
            "attention_mask": attention_mask, # Tensor indicating valid audio parts
            "labels": labels                 # List of integer token IDs
        }
    



    


# import torch
# import torchaudio

# class AudioSpeechRecognitionDataset:
#     """
#     A dataset class for audio speech recognition tasks.

#     Args:
#         data (list): The dataset containing audio paths and transcription columns.
#         processor (Wav2Vec2Processor): The processor to preprocess the audio data.
#         config (object): Configuration object containing dataset parameters.

#     Attributes:
#         data (list): The dataset containing audio paths and transcription columns.
#         processor (Wav2Vec2Processor): The processor to preprocess the audio data.
#         config (object): Configuration object containing dataset parameters.
#         audio_column (str): The name of the column containing audio file paths.
#         transcription_column (str): The name of the column containing transcriptions.

#     Methods:
#         __len__(): Returns the number of samples in the dataset.
#         __getitem__(item): Returns a dictionary containing processed audio input and transcriptions for the given item index.
#     """

#     def __init__(self, data, processor, config):
#         self.data = data
#         self.processor = processor
#         self.config = config
#         self.audio_column = self.config.audio_column
#         self.text_column = self.config.text_column

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         audio_path = self.data[item][self.audio_column]
#         text = self.data[item][self.text_column]

#         # Load audio file
#         waveform, sample_rate = torchaudio.load(audio_path)

#         # Process audio
#         inputs = self.processor(
#             waveform.squeeze(),
#             sampling_rate=sample_rate,
#             return_tensors="pt",
#             padding="longest",
#         )

#         input_values = inputs.input_values.squeeze()
#         attention_mask = inputs.attention_mask.squeeze()

#         return {
#             "input_values": input_values,
#             "attention_mask": attention_mask,
#             "labels": text,
#         } 