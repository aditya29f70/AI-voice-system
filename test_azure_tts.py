import os
from dotenv import load_dotenv

load_dotenv()

"""
test_azure_tts.py — standalone sanity check for Azure Speech (F0 free tier)
before wiring it into the live call pipeline.

Install: pip install azure-cognitiveservices-speech
"""

import azure.cognitiveservices.speech as speechsdk

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")       # Azure Portal -> your resource -> Keys and Endpoint
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")        # whatever region you picked when creating the resource

# A few neural voices to try across the languages you need.
VOICES = {
    "hindi": "hi-IN-SwaraNeural",
    "telugu": "te-IN-ShrutiNeural",
    "english": "en-IN-NeerjaNeural",   # Indian-English voice — tends to land better on this audience
}

TEST_LINES = {
    "hindi": "नमस्ते, मैं ElevateBox से बात कर रहा हूँ।",
    "telugu": "నమస్తే, నేను ElevateBox నుండి మాట్లాడుతున్నాను.",
    "english": "Hi, I'm calling from ElevateBox about your e-commerce website.",
}


def synthesize_to_file(text: str, voice_name: str, output_path: str) -> None:
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = voice_name

    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"[ok] wrote {output_path}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"[error] canceled: {details.reason} — {details.error_details}")


if __name__ == "__main__":
    for lang, voice in VOICES.items():
        synthesize_to_file(TEST_LINES[lang], voice, f"test_{lang}.wav")