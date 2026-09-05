import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
CHUNK_SIZE = 3200

print("Speak now...")

for _ in range(25):

    audio = sd.rec(
        CHUNK_SIZE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    chunk = audio.flatten()

    amplitude = np.max(np.abs(chunk))
    rms = np.sqrt(np.mean(chunk ** 2))

    print(
        f"Amplitude={amplitude:.5f} | "
        f"RMS={rms:.5f}"
    )