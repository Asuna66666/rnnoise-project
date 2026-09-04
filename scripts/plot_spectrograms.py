from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


# ============================================================
# FILE PATHS
# ============================================================

CLEAN_FILE = Path("audio/prepared/clean.wav")
NOISY_FILE = Path("audio/noisy/car_0db.wav")
DENOISED_FILE = Path("audio/denoised/car_0db_denoised.wav")

RESULTS_DIR = Path("results/spectrograms")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file_path):
    """
    Audio file уншина.

    Хэрэв stereo байвал mono болгож хөрвүүлнэ.
    """

    audio, sample_rate = sf.read(file_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sample_rate


# ============================================================
# SPECTROGRAM FUNCTION
# ============================================================

def plot_spectrogram(
    audio,
    sample_rate,
    title,
    output_file,
):
    """
    Reference RNNoise paper-тэй төстэй
    blue -> green -> yellow -> red palette ашиглана.

    Бүх spectrogram дээр ижил dB range ашигласнаар
    clean, noisy, denoised зургуудыг шударга харьцуулна.
    """

    plt.figure(figsize=(11, 5))

    plt.specgram(
        audio,
        NFFT=1024,
        Fs=sample_rate,
        noverlap=768,
        mode="psd",
        scale="dB",

        # Reference зурагтай ойролцоо өнгө
        cmap="jet",

        # Бүх graph дээр ижил intensity scale
        vmin=-100,
        vmax=-20,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)

    # RNNoise paper-ийн example шиг
    # зөвхөн 0 -> 12 kHz хэсгийг харуулна
    plt.ylim(0, 12000)

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


# ============================================================
# LOAD ALL AUDIO
# ============================================================

clean, clean_sr = load_audio(
    CLEAN_FILE
)

noisy, noisy_sr = load_audio(
    NOISY_FILE
)

denoised, denoised_sr = load_audio(
    DENOISED_FILE
)


# ============================================================
# SAMPLE RATE CHECK
# ============================================================

if not (
    clean_sr
    == noisy_sr
    == denoised_sr
):
    raise ValueError(
        "Sample rates do not match."
    )


# ============================================================
# RNNOISE DELAY CORRECTION
# ============================================================

# Өмнөх evaluation дээр RNNoise output
# 480 sample shift-тэй гарсан.
shift = 480

if len(denoised) > shift:
    denoised = denoised[shift:]


# ============================================================
# MATCH AUDIO LENGTHS
# ============================================================

length = min(
    len(clean),
    len(noisy),
    len(denoised),
)

clean = clean[:length]
noisy = noisy[:length]
denoised = denoised[:length]


print(
    "Sample rate:",
    clean_sr,
)

print(
    "Duration:",
    round(
        length / clean_sr,
        2,
    ),
    "seconds",
)


# ============================================================
# FIGURE 1
# CLEAN SPEECH
# ============================================================

plot_spectrogram(
    clean,
    clean_sr,
    "Clean Speech",
    RESULTS_DIR / "car_0db_clean.png",
)


# ============================================================
# FIGURE 2
# NOISY SPEECH
# ============================================================

plot_spectrogram(
    noisy,
    noisy_sr,
    "Car Noise at 0 dB SNR",
    RESULTS_DIR / "car_0db_noisy.png",
)


# ============================================================
# FIGURE 3
# RNNOISE OUTPUT
# ============================================================

plot_spectrogram(
    denoised,
    denoised_sr,
    "RNNoise Output at 0 dB SNR",
    RESULTS_DIR / "car_0db_denoised.png",
)


# ============================================================
# FINISH
# ============================================================

print("\nSpectrograms saved to:")

print(
    RESULTS_DIR
)

print("\nCreated:")

print(
    RESULTS_DIR
    / "car_0db_clean.png"
)

print(
    RESULTS_DIR
    / "car_0db_noisy.png"
)

print(
    RESULTS_DIR
    / "car_0db_denoised.png"
)


# Бүх figure-ийг зэрэг харуулна
plt.show()