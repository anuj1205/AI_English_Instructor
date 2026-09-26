from faster_whisper import WhisperModel

print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print("Whisper loaded.")
print("Transcribing audio...\n")

segments, info = model.transcribe(
    "test.wav",
    beam_size=5
)

print(f"Detected language: {info.language}")
print(f"Language probability: {info.language_probability:.2f}")

print("\nYou said:")

for segment in segments:
    print(segment.text.strip())