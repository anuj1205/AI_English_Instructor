import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5

print("Available audio devices:\n")
print(sd.query_devices())

print("\n" + "=" * 50)
print("Recording for 5 seconds...")
print("Speak normally into your microphone.")
print("=" * 50)

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

volume = np.max(np.abs(audio))

print("\nRecording finished.")
print(f"Maximum microphone level: {volume:.4f}")