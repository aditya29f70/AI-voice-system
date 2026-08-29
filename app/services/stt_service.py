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
