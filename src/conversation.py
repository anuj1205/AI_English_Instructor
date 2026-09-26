import sounddevice as sd
import soundfile as sf
import requests
from faster_whisper import WhisperModel


# ==============================
# Configuration
# ==============================

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_FILE = "conversation.wav"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"


# ==============================
# Load Whisper
# ==============================

print("Loading Whisper...")

whisper_model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print("Whisper ready.")


# ==============================
# Record microphone
# ==============================

def record_audio():

    print("\nSpeak now...")

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


# ==============================
# Speech → Text
# ==============================

def transcribe_audio():

    segments, info = whisper_model.transcribe(
        AUDIO_FILE,
        beam_size=5
    )

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()


# ==============================
# Ask Qwen
# ==============================

def ask_ai(message):

    data = {
        "model": OLLAMA_MODEL,
        "messages": [
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


# ==============================
# Main conversation
# ==============================

print("\n================================")
print("      AI English Instructor")
print("================================")

record_audio()

text = transcribe_audio()

print("\nYou:")
print(text)

if text:

    response = ask_ai(text)

    print("\nAI:")
    print(response)

else:

    print("\nNo speech detected.")