import os
import sys
from pathlib import Path

# ============================================================
# CUDA DLL PATH
# ============================================================

venv_path = Path(sys.prefix)

cublas_path = (
    venv_path
    / "Lib"
    / "site-packages"
    / "nvidia"
    / "cublas"
    / "bin"
)

cudnn_path = (
    venv_path
    / "Lib"
    / "site-packages"
    / "nvidia"
    / "cudnn"
    / "bin"
)

for dll_path in [cublas_path, cudnn_path]:

    if dll_path.exists():

        os.add_dll_directory(str(dll_path))

        os.environ["PATH"] = (
            str(dll_path)
            + os.pathsep
            + os.environ.get("PATH", "")
        )


# ============================================================
# Other imports
# ============================================================

import sounddevice as sd
import soundfile as sf
import requests
import subprocess
import winsound

from faster_whisper import WhisperModel

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

AUDIO_FILE = "user_audio.wav"
RESPONSE_FILE = "ai_response.wav"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"

PIPER_VOICE = "en_US-lessac-medium"


# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper...")

whisper_model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print("Whisper ready.")


# ============================================================
# RECORD USER
# ============================================================

def record_audio():

    print("\n🎤 Speak now...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    sf.write(
        AUDIO_FILE,
        audio,
        SAMPLE_RATE
    )

    print("Recording finished.")


# ============================================================
# WHISPER
# ============================================================

def transcribe_audio():

    segments, info = whisper_model.transcribe(
        AUDIO_FILE,
        beam_size=5
    )

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()


# ============================================================
# QWEN
# ============================================================

def ask_ai(message):

    data = {
        "model": OLLAMA_MODEL,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an English speaking instructor. "
                    "Help the user improve spoken English. "
                    "Keep responses conversational and concise. "
                    "Correct important grammar mistakes naturally."
                )
            },
            {
                "role": "user",
                "content": message
            }
        ],

        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=120
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# ============================================================
# PIPER TTS
# ============================================================

def speak(text):

    print("\n🔊 AI is speaking...")

    # Use the Python interpreter from the active .venv
    piper_python = sys.executable

    print(f"Piper Python: {piper_python}")

    command = [
        piper_python,
        "-m",
        "piper",
        "-m",
        PIPER_VOICE,
        "-f",
        RESPONSE_FILE,
        "--",
        text
    ]

    subprocess.run(
        command,
        check=True
    )

    winsound.PlaySound(
        RESPONSE_FILE,
        winsound.SND_FILENAME
    )

# ============================================================
# MAIN
# ============================================================

print("\n")
print("=" * 50)
print("        AI ENGLISH INSTRUCTOR")
print("=" * 50)

record_audio()

user_text = transcribe_audio()

print("\nYou:")
print(user_text)

if user_text:

    ai_response = ask_ai(user_text)

    print("\nAI:")
    print(ai_response)

    speak(ai_response)

else:

    print("\nNo speech detected.")