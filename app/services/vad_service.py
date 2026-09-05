import torch
import numpy as np

from silero_vad import load_silero_vad, VADIterator


class VADService:

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
    ):

        self.sample_rate = sample_rate

        self.model = load_silero_vad()

        self.vad_iterator = VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
        )

    def process(
        self,
        audio_chunk: np.ndarray
    ):

        audio = torch.from_numpy(
            audio_chunk
        ).float()

        result = self.vad_iterator(
            audio
        )

        return result

    def reset(self):

        self.vad_iterator.reset_states()