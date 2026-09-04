from pathlib import Path
import subprocess


CLEAN_WAV = Path("audio/prepared/clean.wav")
TEMP_DIR = Path("audio/temp")
OUTPUT_DIR = Path("audio/denoised")

RNNOISE_EXE = Path("rnnoise/examples/rnnoise_demo.exe")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_raw = TEMP_DIR / "clean_input.raw"
output_raw = TEMP_DIR / "clean_output.raw"

output_wav = OUTPUT_DIR / "clean_denoised.wav"


# WAV -> RAW
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i", str(CLEAN_WAV),
        "-ar", "48000",
        "-ac", "1",
        "-f", "s16le",
        str(input_raw),
    ],
    check=True,
)


# RNNoise
subprocess.run(
    [
        str(RNNOISE_EXE),
        str(input_raw),
        str(output_raw),
    ],
    check=True,
)


# RAW -> WAV
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-f", "s16le",
        "-ar", "48000",
        "-ac", "1",
        "-i", str(output_raw),
        str(output_wav),
    ],
    check=True,
)


print("Created:", output_wav)