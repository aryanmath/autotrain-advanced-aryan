# from transformers import Wav2Vec2Processor

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

from transformers import Wav2Vec2Processor
import torchaudio               


class AudioSpeechRecognitionPreprocessor:
    """
    A preprocessor class for audio speech recognition tasks.

    Args:
        processor (Wav2Vec2Processor): The processor to preprocess the audio data.

    Attributes:
        processor (Wav2Vec2Processor): The processor to preprocess the audio data.

    Methods:
        process(audio_path): Processes the audio file and returns input values and attention mask.
    """

    def __init__(self, processor):
        self.processor = processor

    def process(self, audio_path):
        """Processes the audio file and returns input values and attention mask."""
        waveform, sample_rate = torchaudio.load(audio_path)
        inputs = self.processor(
            waveform.squeeze(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="longest",
        )
        return inputs.input_values.squeeze(), inputs.attention_mask.squeeze()
