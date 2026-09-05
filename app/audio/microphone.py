import sounddevice as sd
import numpy as np


class Microphone:

    def __init__(
            self,
            sample_rate: int=16000,
            chunk_ms: int=100,
    ):
        self.sample_rate= sample_rate
        self.chunk_ms= chunk_ms


        self.chunk_size= int(
            sample_rate*chunk_ms/1000
        )

    def record_chunk(self)-> np.ndarray:

        audio= sd.rec(
            self.chunk_size,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        return audio.flatten()