import os
import math
import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg

# Set ffmpeg path
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally', 'right', 'well']


class SpeechAnalyzer:

    def __init__(self, file_path):
        self.file_path = file_path
        self.wav_path = self._convert_to_wav()

    def _convert_to_wav(self):
        """Convert input file to WAV."""
        base, ext = os.path.splitext(self.file_path)
        wav_path = base + ".wav"

        if not os.path.exists(wav_path):
            audio = AudioSegment.from_file(self.file_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")

        return wav_path

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