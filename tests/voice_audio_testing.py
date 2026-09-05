import sounddevice as sd
import numpy as np

print(sd.query_devices())

sample_rate = 16000
duration = 5

print("Speak now...")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

print("Recording finished.")
print("Maximum amplitude:", np.max(np.abs(audio)))
print("Average amplitude:", np.mean(np.abs(audio)))