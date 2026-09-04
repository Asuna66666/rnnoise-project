import soundfile as sf
from pathlib import Path

files = [
    "audio/prepared/clean.wav",
    "audio/prepared/babble_noise.wav",
    "audio/prepared/car_noise.wav",
    "audio/prepared/street_noise.wav",
]

for file in files:
    audio, sr = sf.read(file)

    duration = len(audio) / sr

    print(f"\n{Path(file).name}")
    print("Sample rate:", sr)
    print("Duration:", round(duration, 2), "seconds")
    print("Shape:", audio.shape)