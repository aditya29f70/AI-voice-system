import os
from app.services.tts_service import synthesize_to_file
import winsound  # built into Python on Windows — no pip install needed

# A few neural voices to try across the languages you need.
VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "en": "en-IN-NeerjaNeural",   # Indian-English voice — tends to land better on this audience
}


def text_to_speech(state):
    lang = state["language"]
    output_path = os.path.join(os.getcwd(), f"response_{lang}.wav")

    # already inside an async context (this node itself is async), so just
    # await the coroutine directly — asyncio.run() here would crash with
    # "cannot be called from a running event loop"
    synthesize_to_file(state["current_response"], VOICES[lang], output_path)

    # Local playback for testing only — this plays through YOUR machine's
    # speakers so you can sanity-check the voice. It does NOT play into the
    # actual phone call. Real call audio has to stream back through
    # Twilio's bidirectional WebSocket instead — separate piece, not this.
    winsound.PlaySound(output_path, winsound.SND_FILENAME)

    return {}
    