import numpy as np
import soundfile as sf

from pesq import pesq
from scipy.signal import correlate, resample_poly


CLEAN_FILE = "audio/prepared/clean.wav"
DENOISED_FILE = "audio/denoised/clean_denoised.wav"

TARGET_SR = 16000


def align_signals(reference, test, max_shift=2000):

    length = min(
        len(reference),
        len(test),
    )

    reference = reference[:length]
    test = test[:length]

    search_length = min(
        length,
        48000 * 5,
    )

    ref_search = (
        reference[:search_length]
        - np.mean(reference[:search_length])
    )

    test_search = (
        test[:search_length]
        - np.mean(test[:search_length])
    )

    correlation = correlate(
        test_search,
        ref_search,
        mode="full",
        method="fft",
    )

    lags = np.arange(
        -len(ref_search) + 1,
        len(test_search),
    )

    valid = (
        (lags >= -max_shift)
        & (lags <= max_shift)
    )

    best_shift = int(
        lags[valid][
            np.argmax(correlation[valid])
        ]
    )

    if best_shift > 0:

        reference = reference[:-best_shift]
        test = test[best_shift:]

    elif best_shift < 0:

        reference = reference[-best_shift:]
        test = test[:best_shift]

    length = min(
        len(reference),
        len(test),
    )

    return (
        reference[:length],
        test[:length],
        best_shift,
    )


def resample_16k(audio):

    return resample_poly(
        audio,
        TARGET_SR,
        48000,
    )


clean, sr1 = sf.read(CLEAN_FILE)
denoised, sr2 = sf.read(DENOISED_FILE)

if sr1 != 48000 or sr2 != 48000:
    raise ValueError("Expected 48 kHz audio.")


aligned_clean, aligned_denoised, shift = (
    align_signals(
        clean,
        denoised,
    )
)


clean_16k = resample_16k(aligned_clean)
denoised_16k = resample_16k(aligned_denoised)

length = min(
    len(clean_16k),
    len(denoised_16k),
)

clean_16k = clean_16k[:length]
denoised_16k = denoised_16k[:length]


# Clean compared with itself
clean_pesq = pesq(
    16000,
    clean_16k,
    clean_16k,
    "wb",
)


# Clean passed through RNNoise
rnnoise_clean_pesq = pesq(
    16000,
    clean_16k,
    denoised_16k,
    "wb",
)


print("Shift:", shift)
print("Clean PESQ:", round(clean_pesq, 3))
print(
    "RNNoise on clean PESQ:",
    round(rnnoise_clean_pesq, 3),
)