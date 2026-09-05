import queue
import numpy as np
import sounddevice as sd
import torch

from silero_vad import load_silero_vad, VADIterator


class AdaptiveRecorder:
    """
    Real-time microphone recorder using:
        sounddevice.InputStream
            ↓
        Silero VADIterator
            ↓
        speech start/end detection

    Returns:
        np.ndarray containing one complete speech utterance
    """

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512  # Required by Silero VAD at 16 kHz

    # VAD
    SPEECH_THRESHOLD = 0.5
    END_SILENCE_MS = 1400 # 800(good for fast conversation)
    SPEECH_PAD_MS = 100

    # Utterance control
    MIN_SPEECH_MS = 150
    MAX_UTTERANCE_MS = 15000

    # Audio before VAD detects speech.
    # Helps prevent losing the first word.
    PRE_BUFFER_MS = 400

    def __init__(self, device=None):
        self.device = device

        print("Loading Silero VAD...")

        self.vad_model = load_silero_vad()

        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=self.SPEECH_THRESHOLD,
            sampling_rate=self.SAMPLE_RATE,
            min_silence_duration_ms=self.END_SILENCE_MS,
            speech_pad_ms=self.SPEECH_PAD_MS,
        )

        print("VAD loaded.")

        # Audio chunks arrive here from sounddevice callback.
        self.audio_queue = queue.Queue()

        self.stream = None

    # ---------------------------------------------------------
    # Sounddevice callback
    # ---------------------------------------------------------

    def _audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):
        """
        Called continuously by sounddevice.

        IMPORTANT:
        Do not run VAD or Whisper here.
        Just copy the audio and put it in the queue.
        """

        if status:
            print(f"\nAudio status: {status}")

        # Copy because sounddevice reuses the callback buffer.
        chunk = indata[:, 0].copy()

        self.audio_queue.put(chunk)

    # ---------------------------------------------------------
    # Start microphone
    # ---------------------------------------------------------

    def _start_stream(self):
        self.stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.CHUNK_SIZE,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )

        self.stream.start()

    # ---------------------------------------------------------
    # Stop microphone
    # ---------------------------------------------------------

    def _stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    # ---------------------------------------------------------
    # Clear queue
    # ---------------------------------------------------------

    def _clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    # ---------------------------------------------------------
    # Record one utterance
    # ---------------------------------------------------------

    def record_utterance(self):
        """
        Waits for speech, records until silence,
        and returns the complete utterance.

        Returns None if no speech is detected.
        """

        print("\n Listening...")

        self._clear_queue()

        # Reset Silero's internal state.
        self.vad_iterator.reset_states()

        # -----------------------------------------------------
        # Pre-buffer
        # -----------------------------------------------------

        chunk_ms = (
            self.CHUNK_SIZE
            / self.SAMPLE_RATE
            * 1000
        )

        pre_buffer_chunks = max(
            1,
            int(self.PRE_BUFFER_MS / chunk_ms)
        )

        pre_buffer = []

        # -----------------------------------------------------
        # State
        # -----------------------------------------------------

        speech_started = False

        utterance_chunks = []

        speech_ms = 0

        total_utterance_ms = 0

        try:
            self._start_stream()

            while True:

                # ---------------------------------------------
                # Get next audio chunk
                # ---------------------------------------------

                try:
                    chunk = self.audio_queue.get(
                        timeout=1.0
                    )
                except queue.Empty:
                    continue

                # Make sure shape is correct.
                chunk = np.asarray(
                    chunk,
                    dtype=np.float32
                ).flatten()

                # Silero requires exactly 512 samples at 16k.
                if len(chunk) != self.CHUNK_SIZE:
                    continue

                # ---------------------------------------------
                # Calculate basic audio level
                # ---------------------------------------------

                rms = float(
                    np.sqrt(
                        np.mean(
                            np.square(chunk)
                        )
                    )
                )

                # ---------------------------------------------
                # VAD
                # ---------------------------------------------

                audio_tensor = torch.from_numpy(chunk)

                with torch.no_grad():
                    vad_result = self.vad_iterator(
                        audio_tensor
                    )

                # ---------------------------------------------
                # Speech starts
                # ---------------------------------------------

                if not speech_started:

                    # Keep recent audio.
                    pre_buffer.append(chunk)

                    if len(pre_buffer) > pre_buffer_chunks:
                        pre_buffer.pop(0)

                    # VAD detected speech.
                    if vad_result and "start" in vad_result:

                        speech_started = True

                        print(
                            f"\n Speech started "
                            f"(RMS={rms:.5f})"
                        )

                        # Include audio before VAD detection.
                        utterance_chunks.extend(
                            pre_buffer
                        )

                        speech_ms = (
                            len(utterance_chunks)
                            * chunk_ms
                        )

                        total_utterance_ms = speech_ms

                        pre_buffer.clear()

                    continue

                # ---------------------------------------------
                # Speech already started
                # ---------------------------------------------

                utterance_chunks.append(chunk)

                total_utterance_ms += chunk_ms

                # ---------------------------------------------
                # VAD events
                # ---------------------------------------------

                if vad_result:

                    # Speech started again after temporary silence.
                    if "start" in vad_result:

                        print(
                            "\n Speech resumed"
                        )

                    # Speech ended.
                    elif "end" in vad_result:

                        speech_ms = (
                            len(utterance_chunks)
                            * chunk_ms
                        )

                        print(
                            f"\n Speech ended "
                            f"({speech_ms / 1000:.2f}s)"
                        )

                        # -------------------------------------
                        # Ignore tiny noise
                        # -------------------------------------

                        if speech_ms < self.MIN_SPEECH_MS:

                            print(
                                " Too short. "
                                "Ignoring noise."
                            )

                            speech_started = False

                            utterance_chunks.clear()

                            speech_ms = 0

                            total_utterance_ms = 0

                            pre_buffer.clear()

                            self.vad_iterator.reset_states()

                            print(
                                "\n Listening..."
                            )

                            continue

                        break

                # ---------------------------------------------
                # Maximum utterance duration
                # ---------------------------------------------

                if (
                    total_utterance_ms
                    >= self.MAX_UTTERANCE_MS
                ):

                    print(
                        "\n Maximum utterance "
                        "duration reached."
                    )

                    break

                # ---------------------------------------------
                # Debug display
                # ---------------------------------------------

                print(
                    f"\rRMS={rms:.5f} | "
                    f"utterance="
                    f"{total_utterance_ms / 1000:.1f}s",
                    end="",
                    flush=True,
                )

        except KeyboardInterrupt:

            print(
                "\n\n Recording interrupted."
            )

            raise

        finally:

            self._stop_stream()

            self.vad_iterator.reset_states()

        # -----------------------------------------------------
        # No audio
        # -----------------------------------------------------

        if not utterance_chunks:
            return None

        # -----------------------------------------------------
        # Combine chunks
        # -----------------------------------------------------

        audio = np.concatenate(
            utterance_chunks
        ).astype(np.float32)

        print(
            f"\n Captured "
            f"{len(audio) / self.SAMPLE_RATE:.2f}s "
            f"of audio."
        )

        return audio

    # ---------------------------------------------------------
    # Save audio
    # ---------------------------------------------------------

    def save_audio(
        self,
        audio: np.ndarray,
        path: str = "audio.wav",
    ):
        """
        Save numpy audio to WAV.
        """

        import soundfile as sf

        sf.write(
            path,
            audio,
            self.SAMPLE_RATE,
        )

        print(
            f" Audio saved to {path}"
        )