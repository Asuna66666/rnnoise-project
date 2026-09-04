import matplotlib.pyplot as plt


# ============================================================
# X AXIS
# ============================================================

snr_labels = [
    "0",
    "5",
    "10",
    "15",
    "20",
    "clean",
]


# ============================================================
# MEASURED PESQ DATA
# ============================================================

data = {
    "Babble noise": {
        "RNNoise": [
            1.186,
            1.351,
            1.497,
            1.665,
            1.879,
            2.218,
        ],
        "Noisy": [
            1.144,
            1.272,
            1.545,
            2.021,
            2.626,
            4.644,
        ],
    },

    "Car noise": {
        "RNNoise": [
            1.445,
            1.569,
            1.731,
            1.903,
            2.073,
            2.218,
        ],
        "Noisy": [
            1.206,
            1.358,
            1.648,
            2.130,
            2.795,
            4.644,
        ],
    },

    "Street noise": {
        "RNNoise": [
            1.303,
            1.463,
            1.609,
            1.809,
            1.995,
            2.218,
        ],
        "Noisy": [
            1.155,
            1.384,
            1.764,
            2.315,
            3.009,
            4.644,
        ],
    },
}


# ============================================================
# CREATE 3 PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14, 5),
    sharey=True,
)


for ax, (noise_name, values) in zip(
    axes,
    data.items(),
):

    # RNNoise
    ax.plot(
        snr_labels,
        values["RNNoise"],
        marker="s",
        linewidth=2,
        label="RNNoise",
    )

    # Noisy
    ax.plot(
        snr_labels,
        values["Noisy"],
        marker="v",
        linewidth=2,
        label="Noisy",
    )

    ax.set_title(
        noise_name,
        fontsize=13,
    )

    ax.set_xlabel(
        "SNR (dB)",
    )

    ax.set_ylabel(
        "PESQ MOS-LQO",
    )

    ax.set_ylim(
        1.0,
        4.8,
    )

    ax.grid(
        True,
        alpha=0.4,
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        frameon=False,
    )


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "PESQ MOS-LQO Quality Evaluation",
    fontsize=15,
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0.10, 1, 0.93]
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "results/pesq_comparison_final.png",
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# SHOW
# ============================================================

plt.show()