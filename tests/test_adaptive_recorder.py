from app.audio.adaptive_recorder import AdaptiveRecorder


def main():

    recorder = AdaptiveRecorder()

    audio = recorder.record_utterance()

    if audio is None:
        print("No speech detected.")
        return

    recorder.save_audio(
        audio,
        "test_utterance.wav"
    )

    print("Done.")


if __name__ == "__main__":
    main()