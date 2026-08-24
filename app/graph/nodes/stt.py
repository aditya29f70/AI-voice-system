from app.services.stt_service import STTService
from langchain_core.messages import HumanMessage


def speech_to_text(state):
    audio= state["current_audio"]

    stt= STTService()

    transcript= stt.transcribe(audio)

    return {
        "messages": [HumanMessage(content=transcript)],
        "current_transcript": transcript
    }