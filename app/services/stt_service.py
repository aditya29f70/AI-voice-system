import os
import sys

venv_path = sys.prefix

cublas_path = os.path.join(
    venv_path,
    "Lib",
    "site-packages",
    "nvidia",
    "cublas",
    "bin"
)

cudnn_path = os.path.join(
    venv_path,
    "Lib",
    "site-packages",
    "nvidia",
    "cudnn",
    "bin"
)

# Add CUDA DLL folders to PATH BEFORE importing faster_whisper
os.environ["PATH"] = (
    cublas_path
    + os.pathsep
    + cudnn_path
    + os.pathsep
    + os.environ["PATH"]
)

if sys.platform == "win32":
    os.add_dll_directory(cublas_path)
    os.add_dll_directory(cudnn_path)

print("cuBLAS:", cublas_path)
print("cuDNN:", cudnn_path)


import tempfile

from scipy.io.wavfile import write

from faster_whisper import WhisperModel


class STTService:

    def __init__(self):

        self.model = WhisperModel(
            "small",
            device="cuda",
            compute_type="float16",
        )

    def transcribe(
        self,
        audio,
        sample_rate=16000,
    ):

        temp_path = None

        try:

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp:

                temp_path = temp.name

            write(
                temp_path,
                sample_rate,
                audio,
            )

            segments, info = self.model.transcribe(
                temp_path,
                beam_size=5,
                vad_filter=True,
            )

            text = " ".join(
                segment.text
                for segment in segments
            )

            return text.strip()

        finally:

            if temp_path and os.path.exists(
                temp_path
            ):

                os.remove(temp_path)