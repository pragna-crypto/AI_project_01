import os
import re
import wave
import math
import subprocess
import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg

# Get the ffmpeg executable path from imageio_ffmpeg
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Add the ffmpeg binary directory to PATH so pydub can find it for all subprocess calls.
# Without this, pydub raises [WinError 2] on Windows when trying to run ffmpeg.
_ffmpeg_dir = os.path.dirname(_FFMPEG_EXE)
if _ffmpeg_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

# Set BOTH converter and ffprobe for pydub.
AudioSegment.converter = _FFMPEG_EXE
AudioSegment.ffprobe = _FFMPEG_EXE

FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally', 'right', 'well']

# Audio formats that need conversion to WAV before processing
SUPPORTED_INPUT_FORMATS = ['.wav', '.weba', '.webm', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.mp4']

# Mapping from file extension to the format string pydub/ffmpeg expects
_EXT_TO_FORMAT = {
    '.wav': 'wav',
    '.weba': 'webm',
    '.webm': 'webm',
    '.mp3': 'mp3',
    '.ogg': 'ogg',
    '.flac': 'flac',
    '.m4a': 'mp4',
    '.aac': 'aac',
    '.mp4': 'mp4',
}


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

        # Determine the input format hint for pydub/ffmpeg
        input_format = _EXT_TO_FORMAT.get(ext)

        # --- Primary method: pydub with explicit format hint ---
        converted = False
        pydub_error = None
        try:
            print(f"[SpeechAnalyzer] Converting '{os.path.basename(self.file_path)}' ({ext}) -> WAV via pydub...")
            audio = AudioSegment.from_file(self.file_path, format=input_format)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # 16-bit PCM
            audio.export(wav_path, format="wav")
            converted = True
            print(f"[SpeechAnalyzer] Conversion complete -> '{os.path.basename(wav_path)}'")
        except Exception as e:
            pydub_error = e
            print(f"[SpeechAnalyzer] pydub conversion failed: {e}")

        # --- Fallback method: call ffmpeg directly via subprocess ---
        if not converted:
            try:
                print(f"[SpeechAnalyzer] Retrying with direct ffmpeg subprocess...")
                cmd = [
                    _FFMPEG_EXE,
                    '-y',               # overwrite output
                    '-i', self.file_path,
                    '-ac', '1',         # mono
                    '-ar', '16000',     # 16 kHz
                    '-sample_fmt', 's16',  # 16-bit PCM
                    wav_path,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg exited with code {result.returncode}: {result.stderr[-500:]}")
                converted = True
                print(f"[SpeechAnalyzer] Direct ffmpeg conversion complete -> '{os.path.basename(wav_path)}'")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to convert '{os.path.basename(self.file_path)}' to WAV. "
                    f"pydub error: {pydub_error}  |  ffmpeg error: {e2}"
                ) from e2

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
        print(f"[SpeechAnalyzer] Transcript: {repr(transcript[:200]) if transcript else '(empty)'}")

        speed_wpm = self._calculate_speed(transcript)
        print(f"[SpeechAnalyzer] Speech speed: {speed_wpm} WPM")

        filler_count = self._count_fillers(transcript)
        print(f"[SpeechAnalyzer] Filler words found: {filler_count}")

        pause_count = self._detect_pauses()
        print(f"[SpeechAnalyzer] Pauses detected: {pause_count}")

        voice_stability = self._calculate_stability()
        print(f"[SpeechAnalyzer] Voice stability: {voice_stability}")

        scores = self._calculate_scores(
            speed_wpm, filler_count, pause_count, voice_stability
        )
        print(f"[SpeechAnalyzer] Scores: {scores}")

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
        """Count filler words using word-boundary matching."""
        if not transcript:
            return 0

        transcript_lower = transcript.lower()
        count = 0
        for word in FILLER_WORDS:
            # Use regex word boundaries so 'well' doesn't match inside 'farewell'
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.findall(pattern, transcript_lower)
            if matches:
                print(f"[SpeechAnalyzer]   Filler '{word}' found {len(matches)} time(s)")
            count += len(matches)

        return count

    def _detect_pauses(self):
        """Detect pauses in speech based on silence gaps."""
        try:
            y, sample_rate = librosa.load(self.wav_path, sr=None)
            duration = len(y) / sample_rate
            print(f"[SpeechAnalyzer] Audio loaded: {duration:.2f}s, sample_rate={sample_rate}")

            # Use top_db=20 for better sensitivity to pauses
            # (lower value = more sensitive to quieter sounds = more pauses detected)
            intervals = librosa.effects.split(y, top_db=20)
            print(f"[SpeechAnalyzer] Speech intervals found: {len(intervals)}")

            # Count gaps between speech intervals that are >= 0.3 seconds
            pause_count = 0
            min_pause_duration = 0.3  # seconds
            for i in range(1, len(intervals)):
                gap_start = intervals[i - 1][1]  # end of previous speech
                gap_end = intervals[i][0]          # start of next speech
                gap_duration = (gap_end - gap_start) / sample_rate
                if gap_duration >= min_pause_duration:
                    pause_count += 1

            print(f"[SpeechAnalyzer] Pauses >= {min_pause_duration}s: {pause_count}")
            return pause_count

        except Exception as e:
            print(f"[SpeechAnalyzer] Pause detection error: {e}")
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