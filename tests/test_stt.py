from app.services.stt_service import STTService


stt= STTService()

text= stt.transcribe("audio.wav")
print("Transcription: ")
print(text)