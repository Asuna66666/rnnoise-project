from pathlib import Path

import numpy as np
import soundfile as sf


SNR_LEVELS = [0, 5, 10, 15, 20]

NOISE_FILES = {
    "babble": "audio/prepared/babble_noise.wav",
    "car": "audio/prepared/car_noise.wav",
    "street": "audio/prepared/street_noise.wav",
}

CLEAN_FILE = "audio/prepared/clean.wav"
OUTPUT_DIR = Path("audio/noisy")


def match_noise_length(noise, target_length):
    """
    Noise богино байвал давтаж,
    урт байвал clean audio-тай ижил урт болтол тайрна.
    """

    if len(noise) >= target_length:
        return noise[:target_length]

    repeats = int(np.ceil(target_length / len(noise)))
    noise = np.tile(noise, repeats)

    return noise[:target_length]


def mix_at_snr(clean, noise, snr_db):
    """
    Clean болон noise signal-ийг хүссэн SNR дээр mix хийнэ.
    """

    clean_rms = np.sqrt(np.mean(clean ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2))

    if noise_rms == 0:
        raise ValueError("Noise audio is silent.")

    # Desired:
    # SNR = 20 * log10(clean_rms / noise_rms)

    desired_noise_rms = clean_rms / (10 ** (snr_db / 20))

    noise_scaled = noise * (desired_noise_rms / noise_rms)

    noisy = clean + noise_scaled

    # Clipping гарахаас хамгаална
    peak = np.max(np.abs(noisy))

    if peak > 1.0:
        noisy = noisy / peak * 0.99

    return noisy


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean, clean_sr = sf.read(CLEAN_FILE)

    print("Clean sample rate:", clean_sr)
    print("Clean duration:", round(len(clean) / clean_sr, 2), "seconds")

    for noise_name, noise_file in NOISE_FILES.items():

        noise, noise_sr = sf.read(noise_file)

        if noise_sr != clean_sr:
            raise ValueError(
                f"{noise_name}: sample rate mismatch "
                f"{noise_sr} != {clean_sr}"
            )

        noise = match_noise_length(
            noise,
            len(clean)
        )

        for snr_db in SNR_LEVELS:

            noisy = mix_at_snr(
                clean,
                noise,
                snr_db
            )

            output_file = (
                OUTPUT_DIR
                / f"{noise_name}_{snr_db}db.wav"
            )

            sf.write(
                output_file,
                noisy,
                clean_sr,
                subtype="PCM_16"
            )

            print("Created:", output_file)


if __name__ == "__main__":
    main()