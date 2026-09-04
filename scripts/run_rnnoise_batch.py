from pathlib import Path
import subprocess

NOISY_DIR = Path("audio/noisy")
DENOISED_DIR = Path("audio/denoised")
TEMP_DIR = Path("audio/temp")

RNNOISE_EXE = Path("rnnoise/examples/rnnoise_demo.exe")

DENOISED_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

for wav_file in sorted(NOISY_DIR.glob("*.wav")):

    name = wav_file.stem

    input_raw = TEMP_DIR / f"{name}_input.raw"
    output_raw = TEMP_DIR / f"{name}_output.raw"

    output_wav = DENOISED_DIR / f"{name}_denoised.wav"

    print(f"\nProcessing: {wav_file.name}")

    # WAV -> 48 kHz mono 16 bit RAW PCM
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(wav_file),
            "-ar", "48000",
            "-ac", "1",
            "-f", "s16le",
            str(input_raw),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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

    # RAW PCM -> WAV
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("Created:", output_wav)

print("\nAll RNNoise processing complete.")