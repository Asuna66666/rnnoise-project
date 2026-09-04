from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate


CLEAN_FILE = Path("audio/prepared/clean.wav")
NOISY_DIR = Path("audio/noisy")
DENOISED_DIR = Path("audio/denoised")


def snr_db(reference, test):
    """
    reference = clean signal
    test = noisy эсвэл denoised signal
    """

    length = min(len(reference), len(test))

    reference = reference[:length]
    test = test[:length]

    error = test - reference

    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(error ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


def align_signals(reference, test, max_shift=2000):
    """
    RNNoise output дээр бага зэрэг delay орсон байж болох тул
    clean reference болон output signal-ийг cross-correlation
    ашиглан align хийнэ.

    max_shift = 2000 samples
    48 kHz дээр ойролцоогоор 41.7 ms.
    """

    length = min(len(reference), len(test))

    reference = reference[:length]
    test = test[:length]

    # Alignment олоход бүтэн 30 секунд хэрэггүй.
    # Эхний 5 секунд хангалттай.
    search_length = min(length, 48000 * 5)

    ref_search = reference[:search_length]
    test_search = test[:search_length]

    # DC offset бага зэрэг нөлөөлөхөөс хамгаална
    ref_search = ref_search - np.mean(ref_search)
    test_search = test_search - np.mean(test_search)

    # FFT based cross-correlation
    correlation = correlate(
        test_search,
        ref_search,
        mode="full",
        method="fft",
    )

    lags = np.arange(
        -len(ref_search) + 1,
        len(test_search)
    )

    # Зөвхөн боломжит жижиг delay дотор хайна
    valid = (lags >= -max_shift) & (lags <= max_shift)

    valid_correlations = correlation[valid]
    valid_lags = lags[valid]

    best_shift = int(
        valid_lags[np.argmax(valid_correlations)]
    )

    # Signal-уудыг shift-ийн дагуу тааруулна
    if best_shift > 0:
        aligned_reference = reference[:-best_shift]
        aligned_test = test[best_shift:]

    elif best_shift < 0:
        aligned_reference = reference[-best_shift:]
        aligned_test = test[:best_shift]

    else:
        aligned_reference = reference
        aligned_test = test

    aligned_length = min(
        len(aligned_reference),
        len(aligned_test),
    )

    return (
        aligned_reference[:aligned_length],
        aligned_test[:aligned_length],
        best_shift,
    )


def main():
    clean, clean_sr = sf.read(CLEAN_FILE)

    print("Clean file:", CLEAN_FILE)
    print("Sample rate:", clean_sr)
    print(
        "Duration:",
        round(len(clean) / clean_sr, 2),
        "seconds",
    )

    results = []

    noisy_files = sorted(NOISY_DIR.glob("*.wav"))

    print("\nSamples to evaluate:", len(noisy_files))

    for noisy_file in noisy_files:

        denoised_file = (
            DENOISED_DIR
            / f"{noisy_file.stem}_denoised.wav"
        )

        if not denoised_file.exists():
            print(
                "Missing denoised file:",
                denoised_file,
            )
            continue

        noisy, noisy_sr = sf.read(noisy_file)
        denoised, denoised_sr = sf.read(
            denoised_file
        )

        if noisy_sr != clean_sr:
            print(
                "Noisy sample rate mismatch:",
                noisy_file.name,
            )
            continue

        if denoised_sr != clean_sr:
            print(
                "Denoised sample rate mismatch:",
                denoised_file.name,
            )
            continue

        # Input noisy signal болон clean нь
        # бидний өөрсдийн controlled mixture учраас
        # шууд SNR хэмжиж болно.
        input_snr = snr_db(
            clean,
            noisy,
        )

        # RNNoise output бага delay оруулж болох тул
        # clean reference-тай align хийнэ.
        aligned_clean, aligned_denoised, shift = (
            align_signals(
                clean,
                denoised,
            )
        )

        output_snr = snr_db(
            aligned_clean,
            aligned_denoised,
        )

        improvement = output_snr - input_snr

        results.append(
            {
                "sample": noisy_file.stem,
                "input_snr": input_snr,
                "output_snr": output_snr,
                "improvement": improvement,
                "shift": shift,
            }
        )

        print(
            "Processed:",
            noisy_file.stem,
        )

    print("\nRNNoise SNR evaluation\n")

    print(
        f"{'Sample':<20}"
        f"{'Input':>10}"
        f"{'Output':>10}"
        f"{'Improve':>12}"
        f"{'Shift':>10}"
    )

    print("-" * 62)

    for result in results:

        print(
            f"{result['sample']:<20}"
            f"{result['input_snr']:>10.2f}"
            f"{result['output_snr']:>10.2f}"
            f"{result['improvement']:>12.2f}"
            f"{result['shift']:>10}"
        )

    if not results:
        print("\nNo valid results were produced.")
        return

    average_input = np.mean(
        [r["input_snr"] for r in results]
    )

    average_output = np.mean(
        [r["output_snr"] for r in results]
    )

    average_improvement = np.mean(
        [r["improvement"] for r in results]
    )

    print("\nOverall averages")
    print("----------------")

    print(
        "Average input SNR:",
        round(average_input, 2),
        "dB",
    )

    print(
        "Average output SNR:",
        round(average_output, 2),
        "dB",
    )

    print(
        "Average improvement:",
        round(average_improvement, 2),
        "dB",
    )


if __name__ == "__main__":
    main()