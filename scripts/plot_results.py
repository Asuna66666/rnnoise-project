from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DATA
# ============================================================

snr_levels = [0, 5, 10, 15, 20]

snr_improvement = {
    "Babble": [6.26, 4.51, 2.07, -1.26, -5.06],
    "Car": [9.80, 7.16, 3.86, -0.07, -4.45],
    "Street": [8.04, 5.59, 2.76, -0.58, -4.63],
}

stoi_input = {
    "Babble": [0.674, 0.790, 0.881, 0.936, 0.965],
    "Car": [0.767, 0.838, 0.893, 0.934, 0.964],
    "Street": [0.800, 0.880, 0.933, 0.963, 0.980],
}

stoi_output = {
    "Babble": [0.801, 0.892, 0.926, 0.943, 0.954],
    "Car": [0.896, 0.920, 0.937, 0.948, 0.956],
    "Street": [0.887, 0.920, 0.940, 0.955, 0.965],
}


# ============================================================
# GRAPH 1
# SNR improvement
# ============================================================

plt.figure(figsize=(10, 6))

for noise_type, values in snr_improvement.items():
    plt.plot(
        snr_levels,
        values,
        marker="o",
        label=noise_type,
    )

plt.axhline(y=0, linewidth=1)

plt.xlabel("Input SNR (dB)")
plt.ylabel("SNR improvement (dB)")
plt.title("RNNoise SNR Improvement")
plt.xticks(snr_levels)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "snr_improvement.png",
    dpi=300,
)


# ============================================================
# GRAPH 2
# STOI before vs after
# ============================================================

average_input = []
average_output = []

for i in range(len(snr_levels)):

    input_values = [
        stoi_input["Babble"][i],
        stoi_input["Car"][i],
        stoi_input["Street"][i],
    ]

    output_values = [
        stoi_output["Babble"][i],
        stoi_output["Car"][i],
        stoi_output["Street"][i],
    ]

    average_input.append(
        np.mean(input_values)
    )

    average_output.append(
        np.mean(output_values)
    )


plt.figure(figsize=(10, 6))

plt.plot(
    snr_levels,
    average_input,
    marker="o",
    label="Before RNNoise",
)

plt.plot(
    snr_levels,
    average_output,
    marker="o",
    label="After RNNoise",
)

plt.xlabel("Input SNR (dB)")
plt.ylabel("STOI score")
plt.title("Speech Intelligibility Before and After RNNoise")
plt.xticks(snr_levels)
plt.ylim(0.6, 1.0)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "stoi_before_after.png",
    dpi=300,
)


# ============================================================
# GRAPH 3
# Average SNR improvement by noise type
# ============================================================

noise_types = [
    "Babble",
    "Car",
    "Street",
]

average_snr_improvement = [
    np.mean(snr_improvement["Babble"]),
    np.mean(snr_improvement["Car"]),
    np.mean(snr_improvement["Street"]),
]

average_stoi_improvement = []

for noise_type in noise_types:

    before = np.array(
        stoi_input[noise_type]
    )

    after = np.array(
        stoi_output[noise_type]
    )

    average_stoi_improvement.append(
        np.mean(after - before)
    )


plt.figure(figsize=(8, 5))

plt.bar(
    noise_types,
    average_snr_improvement,
)

plt.axhline(y=0, linewidth=1)

plt.xlabel("Noise type")
plt.ylabel("Average SNR improvement (dB)")
plt.title("Average RNNoise Improvement by Noise Type")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "average_noise_improvement.png",
    dpi=300,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nAverage SNR improvement")

for noise_type, value in zip(
    noise_types,
    average_snr_improvement,
):
    print(
        noise_type,
        ":",
        round(value, 2),
        "dB",
    )


print("\nAverage STOI improvement")

for noise_type, value in zip(
    noise_types,
    average_stoi_improvement,
):
    print(
        noise_type,
        ":",
        round(value, 3),
    )


print("\nGraphs saved to results/")


# ============================================================
# SHOW ALL 3 WINDOWS
# ============================================================

plt.show()