from pathlib import Path

import numpy as np
import soundfile as sf
from pystoi import stoi
from scipy.signal import correlate


CLEAN_FILE = Path("audio/prepared/clean.wav")
NOISY_DIR = Path("audio/noisy")
DENOISED_DIR = Path("audio/denoised")


def align_signals(reference, test, max_shift=2000):
    """
    RNNoise output-ийн delay-г cross correlation ашиглан олно.
    """

    length = min(len(reference), len(test))

    reference = reference[:length]
    test = test[:length]

    search_length = min(length, 48000 * 5)

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
        len(test_search)
    )

    valid = (
        (lags >= -max_shift)
        & (lags <= max_shift)
    )

    best_shift = int(
        lags[valid][np.argmax(correlation[valid])]
    )

    if best_shift > 0:
        aligned_reference = reference[:-best_shift]
        aligned_test = test[best_shift:]

    elif best_shift < 0:
        aligned_reference = reference[-best_shift:]
        aligned_test = test[:best_shift]

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
        best_shift,
    )


def main():

    clean, clean_sr = sf.read(CLEAN_FILE)

    results = []

    noisy_files = sorted(NOISY_DIR.glob("*.wav"))

    print("Samples:", len(noisy_files))
    print("Sample rate:", clean_sr)

    for noisy_file in noisy_files:

        denoised_file = (
            DENOISED_DIR
            / f"{noisy_file.stem}_denoised.wav"
        )

        if not denoised_file.exists():
            print("Missing:", denoised_file)
            continue

        noisy, noisy_sr = sf.read(noisy_file)

        denoised, denoised_sr = sf.read(
            denoised_file
        )

        if noisy_sr != clean_sr:
            print(
                "Sample rate mismatch:",
                noisy_file.name,
            )
            continue

        if denoised_sr != clean_sr:
            print(
                "Sample rate mismatch:",
                denoised_file.name,
            )
            continue

        # Noisy болон clean нь аль хэдийн aligned
        noisy_length = min(
            len(clean),
            len(noisy),
        )

        clean_for_noisy = clean[:noisy_length]
        noisy = noisy[:noisy_length]

        input_stoi = stoi(
            clean_for_noisy,
            noisy,
            clean_sr,
            extended=False,
        )

        # RNNoise output-ийг clean reference-тай align хийнэ
        aligned_clean, aligned_denoised, shift = (
            align_signals(
                clean,
                denoised,
            )
        )

        output_stoi = stoi(
            aligned_clean,
            aligned_denoised,
            clean_sr,
            extended=False,
        )

        improvement = (
            output_stoi - input_stoi
        )

        results.append(
            {
                "sample": noisy_file.stem,
                "input": input_stoi,
                "output": output_stoi,
                "improvement": improvement,
                "shift": shift,
            }
        )

    print("\nRNNoise STOI evaluation\n")

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
            f"{result['input']:>10.3f}"
            f"{result['output']:>10.3f}"
            f"{result['improvement']:>12.3f}"
            f"{result['shift']:>10}"
        )

    if not results:
        return

    average_input = np.mean(
        [r["input"] for r in results]
    )

    average_output = np.mean(
        [r["output"] for r in results]
    )

    average_improvement = np.mean(
        [r["improvement"] for r in results]
    )

    print("\nOverall averages")
    print("----------------")

    print(
        "Average input STOI:",
        round(average_input, 3),
    )

    print(
        "Average output STOI:",
        round(average_output, 3),
    )

    print(
        "Average improvement:",
        round(average_improvement, 3),
    )


if __name__ == "__main__":
    main()