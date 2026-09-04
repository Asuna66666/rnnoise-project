from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq
from scipy.signal import correlate, resample_poly


CLEAN_FILE = Path("audio/prepared/clean.wav")
NOISY_DIR = Path("audio/noisy")
DENOISED_DIR = Path("audio/denoised")

TARGET_SR = 16000


def resample_to_16k(audio, original_sr):
    """
    PESQ wideband нь 16 kHz audio ашиглана.
    48 kHz -> 16 kHz болгон resample хийнэ.
    """

    if original_sr == TARGET_SR:
        return audio

    return resample_poly(
        audio,
        TARGET_SR,
        original_sr,
    )


def align_signals(reference, test, max_shift=2000):
    """
    RNNoise output-ийн delay-г clean reference-тай align хийнэ.
    """

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

    ref_search = reference[:search_length]
    test_search = test[:search_length]

    ref_search = ref_search - np.mean(ref_search)
    test_search = test_search - np.mean(test_search)

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

        aligned_reference = (
            reference[:-best_shift]
        )

        aligned_test = (
            test[best_shift:]
        )

    elif best_shift < 0:

        aligned_reference = (
            reference[-best_shift:]
        )

        aligned_test = (
            test[:best_shift]
        )

    else:
        aligned_reference = reference
        aligned_test = test

    final_length = min(
        len(aligned_reference),
        len(aligned_test),
    )

    return (
        aligned_reference[:final_length],
        aligned_test[:final_length],
    )


def calculate_pesq(reference, degraded, sr):
    """
    Wideband PESQ MOS LQO.
    """

    reference_16k = resample_to_16k(
        reference,
        sr,
    )

    degraded_16k = resample_to_16k(
        degraded,
        sr,
    )

    length = min(
        len(reference_16k),
        len(degraded_16k),
    )

    reference_16k = reference_16k[:length]
    degraded_16k = degraded_16k[:length]

    return pesq(
        TARGET_SR,
        reference_16k,
        degraded_16k,
        "wb",
    )


def main():

    clean, clean_sr = sf.read(
        CLEAN_FILE
    )

    results = []

    noisy_files = sorted(
        NOISY_DIR.glob("*.wav")
    )

    print(
        "PESQ samples:",
        len(noisy_files),
    )

    for noisy_file in noisy_files:

        denoised_file = (
            DENOISED_DIR
            / f"{noisy_file.stem}_denoised.wav"
        )

        if not denoised_file.exists():
            print(
                "Missing:",
                denoised_file,
            )
            continue

        noisy, noisy_sr = sf.read(
            noisy_file
        )

        denoised, denoised_sr = sf.read(
            denoised_file
        )

        if not (
            noisy_sr
            == denoised_sr
            == clean_sr
        ):
            print(
                "Sample rate mismatch:",
                noisy_file.name,
            )
            continue

        # ------------------------------
        # Noisy PESQ
        # ------------------------------

        noisy_length = min(
            len(clean),
            len(noisy),
        )

        clean_for_noisy = (
            clean[:noisy_length]
        )

        noisy = noisy[:noisy_length]

        noisy_pesq = calculate_pesq(
            clean_for_noisy,
            noisy,
            clean_sr,
        )

        # ------------------------------
        # RNNoise PESQ
        # ------------------------------

        (
            aligned_clean,
            aligned_denoised,
        ) = align_signals(
            clean,
            denoised,
        )

        rnnoise_pesq = calculate_pesq(
            aligned_clean,
            aligned_denoised,
            clean_sr,
        )

        improvement = (
            rnnoise_pesq
            - noisy_pesq
        )

        results.append(
            {
                "sample": noisy_file.stem,
                "noisy": noisy_pesq,
                "rnnoise": rnnoise_pesq,
                "improvement": improvement,
            }
        )

    print(
        "\nPESQ MOS-LQO evaluation\n"
    )

    print(
        f"{'Sample':<20}"
        f"{'Noisy':>10}"
        f"{'RNNoise':>12}"
        f"{'Improve':>12}"
    )

    print("-" * 54)

    for result in results:

        print(
            f"{result['sample']:<20}"
            f"{result['noisy']:>10.3f}"
            f"{result['rnnoise']:>12.3f}"
            f"{result['improvement']:>12.3f}"
        )


if __name__ == "__main__":
    main()