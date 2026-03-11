import os, sys
sys.path.insert(0, r'd:\pragna_repose\AI_project_01\ai_speech_analyzer')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_speech_analyzer.settings')

# Redirect all output to a file
log_file = open(r'd:\pragna_repose\AI_project_01\ai_speech_analyzer\test_result_clean.txt', 'w', encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file

import warnings
warnings.filterwarnings('ignore')

media = r'd:\pragna_repose\AI_project_01\ai_speech_analyzer\media\speeches'

from analyzer.speech_analyzer import SpeechAnalyzer

# Test MP3
print("=" * 50)
print("MP3 TEST")
print("=" * 50)
try:
    mp3_path = os.path.join(media, 'mp3_44100Hz_320kbps_stereo.mp3')
    a = SpeechAnalyzer(mp3_path)
    r = a.analyze()
    for k, v in r.items():
        if k == 'transcript':
            print(f"  {k}: {repr(v[:150]) if v else '(empty)'}")
        elif k == 'scores':
            print(f"  {k}: {dict(v)}")
        else:
            print(f"  {k}: {v}")
    a.cleanup()
    print("MP3 TEST: PASSED")
except Exception as ex:
    import traceback
    traceback.print_exc()
    print("MP3 TEST: FAILED")

# Test WAV
print()
print("=" * 50)
print("WAV TEST")
print("=" * 50)
try:
    wav_path = os.path.join(media, 'recordvoice_1773126152109.wav')
    a2 = SpeechAnalyzer(wav_path)
    r2 = a2.analyze()
    for k, v in r2.items():
        if k == 'transcript':
            print(f"  {k}: {repr(v[:150]) if v else '(empty)'}")
        elif k == 'scores':
            print(f"  {k}: {dict(v)}")
        else:
            print(f"  {k}: {v}")
    a2.cleanup()
    print("WAV TEST: PASSED")
except Exception as ex:
    import traceback
    traceback.print_exc()
    print("WAV TEST: FAILED")

log_file.close()
