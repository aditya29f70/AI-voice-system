from faster_whisper import WhisperModel


class STTService:

    def __init__(self):
        self.model= WhisperModel(
            'small',
            device="cuda",
            compute_type='float16'
        )

    def transcribe(self, audio_path:str)-> str:
        segments, info= self.model.transcribe(
            audio_path,
            beam_size=5
        )

        text= " ".join(segment.text for segment in segments)

        return text.strip()