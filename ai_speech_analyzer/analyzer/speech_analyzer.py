import os
import wave
import math
import tempfile
import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg

# Set ffmpeg path for pydub
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally', 'right', 'well']

# Audio formats that need conversion to WAV before processing
SUPPORTED_INPUT_FORMATS = ['.wav', '.weba', '.webm', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.mp4']


class SpeechAnalyzer:

    def __init__(self, file_path):
        self.file_path = file_path
        self._converted_file = None  # Track temporary converted file for cleanup
        self.wav_path = self._convert_to_wav()

    def _convert_to_wav(self):
        """
        Convert input audio file (e.g. .weba, .webm, .mp3, .ogg) to a valid
        mono 16 kHz WAV file using pydub + ffmpeg, then validate with the wave module.

        If the input is already a valid .wav file, it is still re-exported to
        guarantee the format matches what SpeechRecognition and librosa expect.
        """
        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()

        if ext not in SUPPORTED_INPUT_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Supported formats: {', '.join(SUPPORTED_INPUT_FORMATS)}"
            )

        # Build output path next to the original file
        base, _ = os.path.splitext(self.file_path)
        wav_path = base + "_converted.wav"

        # Always convert to ensure consistent WAV format (mono, 16 kHz, PCM)
        try:
            print(f"[SpeechAnalyzer] Converting '{os.path.basename(self.file_path)}' ({ext}) → WAV …")
            audio = AudioSegment.from_file(self.file_path)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # 16-bit PCM
            audio.export(wav_path, format="wav")
            print(f"[SpeechAnalyzer] Conversion complete → '{os.path.basename(wav_path)}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert '{os.path.basename(self.file_path)}' to WAV: {e}"
            ) from e

        # Validate the converted file using the wave module
        try:
            with wave.open(wav_path, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                n_frames = wf.getnframes()
                print(
                    f"[SpeechAnalyzer] WAV validated — "
                    f"channels={channels}, sample_width={sample_width}, "
                    f"frame_rate={frame_rate}, frames={n_frames}"
                )
                if n_frames == 0:
                    raise RuntimeError("Converted WAV file has 0 frames — the input audio may be empty.")
        except wave.Error as e:
            raise RuntimeError(
                f"Converted file is not a valid WAV: {e}"
            ) from e

        self._converted_file = wav_path  # Mark for cleanup later
        return wav_path

    def cleanup(self):
        """Remove the temporary converted WAV file if it exists."""
        if self._converted_file and os.path.exists(self._converted_file):
            try:
                os.remove(self._converted_file)
                print(f"[SpeechAnalyzer] Cleaned up temporary file: {os.path.basename(self._converted_file)}")
            except OSError:
                pass

    def __del__(self):
        self.cleanup()

    def analyze(self):
        """Run full analysis."""
        transcript = self._transcribe()

        speed_wpm = self._calculate_speed(transcript)
        filler_count = self._count_fillers(transcript)

        pause_count = self._detect_pauses()
        voice_stability = self._calculate_stability()

        scores = self._calculate_scores(
            speed_wpm, filler_count, pause_count, voice_stability
        )

        return {
            "transcript": transcript,
            "speech_speed": speed_wpm,
            "filler_word_count": filler_count,
            "pause_count": pause_count,
            "voice_stability": voice_stability,
            "scores": scores,
            "confidence_score": scores["total"],
        }

    def _transcribe(self):
        recognizer = sr.Recognizer()

        with sr.AudioFile(self.wav_path) as source:
            audio_data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                return ""

    def _calculate_speed(self, transcript):
        """Calculate Words Per Minute."""
        if not transcript:
            return 0

        duration_sec = librosa.get_duration(path=self.wav_path)
        word_count = len(transcript.split())

        if duration_sec == 0:
            return 0

        wpm = (word_count / duration_sec) * 60
        return round(wpm, 2)

    def _count_fillers(self, transcript):
        """Count filler words."""
        if not transcript:
            return 0

        transcript_lower = transcript.lower()
        count = sum(transcript_lower.count(word) for word in FILLER_WORDS)

        return count

    def _detect_pauses(self):
        """Detect pauses in speech."""
        try:
            y, sample_rate = librosa.load(self.wav_path, sr=None)

            intervals = librosa.effects.split(y, top_db=30)

            pause_count = max(0, len(intervals) - 1)

            return pause_count

        except Exception:
            return 0

    def _calculate_stability(self):
        """Measure voice stability using RMS variance."""
        try:
            y, sample_rate = librosa.load(self.wav_path, sr=None)

            rms = librosa.feature.rms(y=y)[0]

            variance = np.var(rms)

            stability = max(0, 100 - (variance * 10000))

            return round(stability, 2)

        except Exception:
            return 0

    def _calculate_scores(self, speed, filler_count, pause_count, voice_stability):

        # Speed score
        if 120 <= speed <= 160:
            speed_score = 25
        elif 100 <= speed < 120 or 160 < speed <= 180:
            speed_score = 20
        elif 80 <= speed < 100 or 180 < speed <= 200:
            speed_score = 10
        else:
            speed_score = 5

        # Filler score
        if filler_count == 0:
            filler_score = 25
        elif filler_count <= 2:
            filler_score = 20
        elif filler_count <= 5:
            filler_score = 15
        elif filler_count <= 10:
            filler_score = 10
        else:
            filler_score = 5

        # Pause score
        duration = librosa.get_duration(path=self.wav_path)

        pauses_per_min = (pause_count / duration) * 60 if duration > 0 else 0

        if 5 <= pauses_per_min <= 15:
            pause_score = 25
        elif pauses_per_min < 5 or 15 < pauses_per_min <= 25:
            pause_score = 20
        else:
            pause_score = 10

        # Stability score
        stability_score = min(25, (voice_stability / 100) * 25)

        total_score = speed_score + filler_score + pause_score + stability_score

        return {
            "speed": round(speed_score, 2),
            "filler": round(filler_score, 2),
            "pause": round(pause_score, 2),
            "stability": round(stability_score, 2),
            "total": round(total_score, 2),
        }