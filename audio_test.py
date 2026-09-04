import soundfile as sf
import numpy as np

# 1. Audio file унших
audio, sample_rate = sf.read("audio/traffic.wav")

# 2. Frame settings
frame_size = int(sample_rate * 0.020)   # 20 ms
hop_size = frame_size // 2              # 50% overlap

# 3. Audio-г frame болгон хуваах
frames = []

for start in range(0, len(audio) - frame_size + 1, hop_size):
    end = start + frame_size
    frame = audio[start:end]
    frames.append(frame)

frames = np.array(frames)

# 4. Results
print("Sample rate:", sample_rate)
print("Frame size:", frame_size, "samples")
print("Hop size:", hop_size, "samples")
print("Number of frames:", len(frames))
print("Frames shape:", frames.shape)

# Эхний frame шалгах
print("\nFirst frame:")
print("Samples:", len(frames[0]))
print("First 10 values:", frames[0][:10])

import matplotlib.pyplot as plt

# 5. Window function
window = np.hanning(frame_size)

# Эхний frame дээр window тавих
windowed_frame = frames[0] * window

# 6. FFT
fft_result = np.fft.rfft(windowed_frame)

# Complex FFT -> magnitude
magnitude = np.abs(fft_result)

# Frequency axis
frequencies = np.fft.rfftfreq(
    frame_size,
    d=1 / sample_rate
)

print("\nFFT information:")
print("FFT bins:", len(fft_result))
print("Magnitude shape:", magnitude.shape)
print("Frequency range:", frequencies[0], "to", frequencies[-1], "Hz")

# 7. Spectrum харах
plt.figure(figsize=(10, 4))
plt.plot(frequencies, magnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency spectrum of first 20 ms frame")
plt.xlim(0, 24000)
plt.tight_layout()
# plt.show()

# 8. 22 frequency bands

num_bands = 22

# Bark scale conversion
def hz_to_bark(f):
    return 6 * np.arcsinh(f / 600)

def bark_to_hz(b):
    return 600 * np.sinh(b / 6)

# 0 Hz -> 24 kHz хүртэл Bark scale дээр 22 band
min_bark = hz_to_bark(0)
max_bark = hz_to_bark(sample_rate / 2)

bark_edges = np.linspace(
    min_bark,
    max_bark,
    num_bands + 1
)

frequency_edges = bark_to_hz(bark_edges)

band_energies = []

power = magnitude ** 2

for i in range(num_bands):
    low = frequency_edges[i]
    high = frequency_edges[i + 1]

    mask = (frequencies >= low) & (frequencies < high)

    energy = np.sum(power[mask])
    band_energies.append(energy)

band_energies = np.array(band_energies)

print("\n22 band information:")
print("Number of bands:", len(band_energies))
print("Band energies:", band_energies)

plt.figure(figsize=(10, 4))
plt.bar(range(num_bands), band_energies)
plt.xlabel("Frequency band")
plt.ylabel("Energy")
plt.title("Energy in 22 Bark-scale bands")
plt.tight_layout()
# plt.show()

from scipy.fft import dct

# 9. Log band energies
log_band_energies = np.log10(band_energies + 1e-12)

# 10. DCT -> BFCC
bfcc = dct(log_band_energies, type=2, norm="ortho")

print("\nBFCC information:")
print("BFCC shape:", bfcc.shape)
print("BFCC values:", bfcc)

plt.figure(figsize=(10, 4))
plt.bar(range(len(bfcc)), bfcc)
plt.xlabel("BFCC coefficient")
plt.ylabel("Value")
plt.title("BFCC features from 22 Bark bands")
plt.tight_layout()
plt.show()

all_bfcc = []

for frame in frames:
    # Window
    windowed_frame = frame * window

    # FFT
    fft_result = np.fft.rfft(windowed_frame)
    magnitude = np.abs(fft_result)
    power = magnitude ** 2

    # 22 Bark band energies
    band_energies = []

    for i in range(num_bands):
        low = frequency_edges[i]
        high = frequency_edges[i + 1]

        mask = (frequencies >= low) & (frequencies < high)

        energy = np.sum(power[mask])
        band_energies.append(energy)

    band_energies = np.array(band_energies)

    # Log energy
    log_band_energies = np.log10(band_energies + 1e-12)

    # DCT -> BFCC
    bfcc = dct(
        log_band_energies,
        type=2,
        norm="ortho"
    )

    all_bfcc.append(bfcc)

all_bfcc = np.array(all_bfcc)

print("\nAll-frame BFCC:")
print("Shape:", all_bfcc.shape)
print("Number of frames:", all_bfcc.shape[0])
print("Features per frame:", all_bfcc.shape[1])