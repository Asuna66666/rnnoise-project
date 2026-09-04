from pathlib import Path
import soundfile as sf


NOISY_DIR = Path("audio/noisy")
DENOISED_DIR = Path("audio/denoised")


noisy_files = sorted(NOISY_DIR.glob("*.wav"))
denoised_files = sorted(DENOISED_DIR.glob("*.wav"))

print("Noisy files:", len(noisy_files))
print("Denoised files:", len(denoised_files))


for noisy_file in noisy_files:

    expected_output = (
        DENOISED_DIR
        / f"{noisy_file.stem}_denoised.wav"
    )

    if not expected_output.exists():
        print("MISSING:", expected_output)
        continue

    noisy, noisy_sr = sf.read(noisy_file)
    denoised, denoised_sr = sf.read(expected_output)

    print(f"\n{noisy_file.stem}")
    print("Noisy duration:",
          round(len(noisy) / noisy_sr, 2))
    print("Denoised duration:",
          round(len(denoised) / denoised_sr, 2))
    print("Sample rates:",
          noisy_sr,
          denoised_sr)