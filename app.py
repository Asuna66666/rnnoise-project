from pathlib import Path
import io
import os
import subprocess
import tempfile
import uuid

import numpy as np
import soundfile as sf
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RNNoise Demo",
    page_icon="🎧",
    layout="centered",
)


# ============================================================
# PATHS
# ============================================================

CLEAN_FILE = Path("audio/prepared/clean.wav")
NOISY_DIR = Path("audio/noisy")
DENOISED_DIR = Path("audio/denoised")


# ============================================================
# DATA
# ============================================================

NOISE_TYPES = {
    "Олон хүний ярианы чимээ": "babble",
    "Машины чимээ": "car",
    "Гудамжны чимээ": "street",
}

SNR_LEVELS = [0, 5, 10, 15, 20]


# ============================================================
# TITLE
# ============================================================

st.title("RNNoise аудионы демо")

st.write(
    "Өөр өөр төрлийн дуу чимээ болон SNR түвшинд "
    "эх аудио болон RNNoise-оор цэвэрлэсэн аудиог харьцуулна. "
)


# ============================================================
# CONTROLS
# ============================================================

noise_label = st.segmented_control(
    "Дуу чимээний төрөл",
    options=list(NOISE_TYPES.keys()),
    default="Машины чимээ",
)

snr = st.segmented_control(
    "Оролтын SNR",
    options=SNR_LEVELS,
    default=0,
    format_func=lambda x: f"{x} dB",
)


noise_key = NOISE_TYPES[noise_label]


# ============================================================
# FILE PATHS
# ============================================================

noisy_file = (
    NOISY_DIR
    / f"{noise_key}_{snr}db.wav"
)

denoised_file = (
    DENOISED_DIR
    / f"{noise_key}_{snr}db_denoised.wav"
)


# ============================================================
# CHECK FILES
# ============================================================

if not noisy_file.exists():
    st.error(
        f"Missing noisy file: {noisy_file}"
    )
    st.stop()

if not denoised_file.exists():
    st.error(
        f"Missing denoised file: {denoised_file}"
    )
    st.stop()

if not CLEAN_FILE.exists():
    st.error(
        f"Missing clean file: {CLEAN_FILE}"
    )
    st.stop()


# ============================================================
# CURRENT TEST
# ============================================================

st.divider()

st.subheader(
    f"{noise_label} дуу чимээ, {snr} dB SNR"
)


# ============================================================
# AUDIO COMPARISON
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.markdown("### Боловсруулаагүй")

    st.audio(
        str(noisy_file),
        format="audio/wav",
    )

    st.caption(
        "Цэвэр яриаг орчны дуу чимээтэй хольсон аудио."
    )


with col2:

    st.markdown("### RNNoise")

    st.audio(
        str(denoised_file),
        format="audio/wav",
    )

    st.caption(
        "Ижил аудиог RNNoise ашиглан боловсруулсны дараах хувилбар."
    )


# ============================================================
# CLEAN REFERENCE
# ============================================================

st.divider()

with st.expander(
    "Цэвэр эх аудиог сонсох"
):

    st.audio(
        str(CLEAN_FILE),
        format="audio/wav",
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.subheader("Юуг анзаарах вэ?")


if snr == 0:

    st.write(
        "0 dB үед яриа болон дуу чимээний хүч ойролцоо байна. "
        "RNNoise орчны чимээг их хэмжээгээр багасгах боловч, "
        "яриа бага зэрэг робот эсвэл металл мэт сонсогдож болно."
    )

elif snr == 5:

    st.write(
        "5 dB үед дуу чимээ мэдэгдэхүйц багасах бөгөөд "
        "0 dB-тай харьцуулахад ярианы гажуудал бага байна."
    )

elif snr == 10:

    st.write(
        "10 dB үед яриа илүү натурал сонсогдохын зэрэгцээ "
        "орчны дуу чимээ мөн мэдэгдэхүйц багасна."
    )

elif snr == 15:

    st.write(
        "15 dB үед эх аудио аль хэдийн харьцангуй цэвэр тул "
        "RNNoise-ийн нэмэлт ашиг бага болж эхэлнэ."
    )

elif snr == 20:

    st.write(
        "20 dB үед эх аудио маш цэвэр байна. "
        "Ийм үед RNNoise-ийн боловсруулалт нь дуу чимээг багасгахаас илүү "
        "ярианы чанарт бага зэрэг өөрчлөлт оруулж болзошгүй."
    )


# ============================================================
# PRESENTATION NOTE
# ============================================================

st.info(
    "Демо үзүүлэхэд санал болгох дараалал: "
    "0 dB -> 10 dB -> 20 dB."
)

# ============================================================
# LIVE MICROPHONE DEMO
# ============================================================

st.divider()

st.header("Өөрийн хоолойгоор туршиж үзэх")

st.write(
    "Микрофоноор яриагаа бичээд RNNoise ашиглан дуу чимээг "
    "багасгасан хувилбартай харьцуулж сонсоно."
)

st.warning(
    "Чихэвч ашиглахыг зөвлөж байна. Speaker ашиглавал "
    "микрофонд feedback орж болно."
)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(
        uuid.uuid4()
    )

LIVE_DIR = (
    Path(tempfile.gettempdir())
    / "rnnoise_demo"
    / st.session_state.session_id
)

LIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if os.name == "nt":
    RNNOISE_EXE = Path(
        "rnnoise/examples/rnnoise_demo.exe"
    )
else:
    RNNOISE_EXE = Path(
        "rnnoise/examples/rnnoise_demo"
    )


# ------------------------------------------------------------
# CONTROLS
# ------------------------------------------------------------

processing_mode = st.segmented_control(
    "Боловсруулалт",
    options=[
        "Боловсруулахгүй",
        "RNNoise",
    ],
    default="RNNoise",
)

noise_mode = st.segmented_control(
    "Нэмэлт дуу чимээ",
    options=[
        "Байхгүй",
        "White noise",
    ],
    default="Байхгүй",
)


if noise_mode == "White noise":
    noise_snr = st.segmented_control(
        "White noise SNR",
        options=[0, 5, 10, 15, 20],
        default=10,
        format_func=lambda x: f"{x} dB",
    )
else:
    noise_snr = None


# SNR тайлбар
if noise_mode == "White noise":
    if noise_snr == 0:
        st.caption(
            "0 dB: Яриа болон noise ойролцоо хүчтэй. "
            "Хамгийн хүнд туршилтын нөхцөл."
        )

    elif noise_snr == 5:
        st.caption(
            "5 dB: Noise хүчтэй хэвээр боловч яриа арай тод."
        )

    elif noise_snr == 10:
        st.caption(
            "10 dB: Noise suppression болон natural speech-ийн "
            "хооронд сайн balance."
        )

    elif noise_snr == 15:
        st.caption(
            "15 dB: Input харьцангуй цэвэр."
        )

    elif noise_snr == 20:
        st.caption(
            "20 dB: Input аль хэдийн маш цэвэр тул "
            "RNNoise-ийн нэмэлт ашиг бага."
        )



# ------------------------------------------------------------
# MICROPHONE RECORDING
# ------------------------------------------------------------

recording = st.audio_input(
    "Микрофоноор яриагаа бичнэ үү"
)


# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------

def add_white_noise(
    audio,
    snr_db,
):
    """
    Original speech дээр сонгосон SNR түвшний
    white noise нэмнэ.
    """

    clean_rms = np.sqrt(
        np.mean(audio ** 2)
    )

    if clean_rms == 0:
        return audio

    white_noise = np.random.normal(
        0,
        1,
        len(audio),
    )

    noise_rms = np.sqrt(
        np.mean(white_noise ** 2)
    )

    desired_noise_rms = (
        clean_rms
        / (10 ** (snr_db / 20))
    )

    white_noise = (
        white_noise
        * (
            desired_noise_rms
            / noise_rms
        )
    )

    noisy = audio + white_noise

    peak = np.max(
        np.abs(noisy)
    )

    if peak > 1:
        noisy = noisy / peak * 0.99

    return noisy


def run_rnnoise(
    input_wav,
    output_wav,
):
    input_raw = (
        LIVE_DIR
        / "input.raw"
    )

    output_raw = (
        LIVE_DIR
        / "output.raw"
    )

    if not RNNOISE_EXE.exists():
        raise FileNotFoundError(
            f"RNNoise executable not found: "
            f"{RNNOISE_EXE}"
        )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "48000",
            "-ac",
            "1",
            "-f",
            "s16le",
            str(input_raw),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        [
            str(RNNOISE_EXE),
            str(input_raw),
            str(output_raw),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "s16le",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-i",
            str(output_raw),
            str(output_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ------------------------------------------------------------
# PROCESS RECORDING
# ------------------------------------------------------------

if recording is not None:

    original_wav = (
        LIVE_DIR
        / "recorded.wav"
    )

    test_wav = (
        LIVE_DIR
        / "recorded_test.wav"
    )

    denoised_wav = (
        LIVE_DIR
        / "recorded_denoised.wav"
    )


    # Browser recording-г WAV болгож хадгална
    original_wav.write_bytes(
        recording.getvalue()
    )


    # Audio унших
    audio, sample_rate = sf.read(
        io.BytesIO(
            recording.getvalue()
        )
    )


    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1,
        )


    # White noise нэмэх эсэх
    if noise_mode == "White noise":

        audio = add_white_noise(
            audio,
            noise_snr,
        )


    # Test audio хадгалах
    sf.write(
        test_wav,
        audio,
        sample_rate,
    )


    st.subheader(
        "Оролтын аудио"
    )

    st.audio(
        str(test_wav)
    )


    # --------------------------------------------------------
    # RNNoise PROCESSING
    # --------------------------------------------------------

    if processing_mode == "RNNoise":

        if st.button(
            "RNNoise ажиллуулах",
            type="primary",
        ):

            with st.spinner(
                "RNNoise боловсруулж байна..."
            ):

                run_rnnoise(
                    test_wav,
                    denoised_wav,
                )

            st.success(
                "Боловсруулалт дууслаа."
            )

            st.subheader(
                "RNNoise боловсруулсны дараа"
            )

            st.audio(
                str(denoised_wav)
            )


    else:

        st.info(
            "Боловсруулалт хийгдээгүй байна."
        )