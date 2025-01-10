import numpy as np
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper
import time

# Generate a synthetic signal (sum of sinusoids + noise)
fs = 128  # Sampling frequency in Hz
duration = 30  # Duration in seconds
t = np.arange(0, duration, 1/fs)  # Time vector
frequencies = [10, 50]  # Frequencies of interest in Hz

# Create a signal: two sinusoids and Gaussian noise
signal = 2 * np.sin(2 * np.pi * frequencies[0] * t) + \
         0.5 * np.sin(2 * np.pi * frequencies[1] * t) + \
         np.random.randn(len(t))

# Set parameters for PSD estimation
nw = 4  # Time-bandwidth product for multitaper method
fmin, fmax = 0, 500  # Frequency range of interest for PSD

# 1. Multitaper Method with adaptive=True
start_time = time.time()
psd_mt_adaptive, freqs_mt_adaptive = psd_array_multitaper(
    signal, sfreq=fs, fmin=fmin, fmax=fmax, bandwidth=nw, adaptive=True, n_jobs=1
)
time_mt_adaptive = time.time() - start_time

# 2. Multitaper Method with adaptive=False
start_time = time.time()
psd_mt_nonadaptive, freqs_mt_nonadaptive = psd_array_multitaper(
    signal, sfreq=fs, fmin=fmin, fmax=fmax, bandwidth=nw, adaptive=False, n_jobs=1
)
time_mt_nonadaptive = time.time() - start_time

# 3. Welch's Method
nperseg = 64  # Segment length
overlap = nperseg // 2  # 50% overlap
start_time = time.time()
freqs_welch, psd_welch = welch(signal, fs=fs, nperseg=nperseg, noverlap=overlap)
time_welch = time.time() - start_time

# Print execution times
print(f"Execution time (Multitaper, adaptive=True): {time_mt_adaptive:.4f} seconds")
print(f"Execution time (Multitaper, adaptive=False): {time_mt_nonadaptive:.4f} seconds")
print(f"Execution time (Welch's method): {time_welch:.4f} seconds")

# Plot the PSD results for comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(freqs_mt_adaptive, psd_mt_adaptive, label='Multitaper (adaptive=True)', alpha=0.7)
plt.plot(freqs_mt_nonadaptive, psd_mt_nonadaptive, label='Multitaper (adaptive=False)', alpha=0.7)
plt.plot(freqs_welch, psd_welch, label="Welch's Method", alpha=0.7)
plt.title("Power Spectral Density Comparison", fontsize=14)
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Power Spectral Density (PSD)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()