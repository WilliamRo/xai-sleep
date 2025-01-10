import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram
from mne.time_frequency import psd_array_multitaper  # Multitaper method from MNE library

# Generate synthetic EEG data for illustration (replace with your EEG data)
# Signal parameters
fs = 256  # Sampling frequency in Hz
duration = 10  # Duration in seconds
frequencies = [10, 20]  # Frequencies in Hz
t = np.arange(0, duration, 1/fs)  # Time vector

# Simulate an EEG-like signal (sum of sinusoids + noise)
eeg_data = 0.5 * np.sin(2 * np.pi * frequencies[0] * t) + \
           0.3 * np.sin(2 * np.pi * frequencies[1] * t) + 0.2 * np.random.randn(len(t))

# --- Method 1: Raw Periodogram ---
freqs_periodogram, psd_periodogram = periodogram(eeg_data, fs)

# --- Method 2: Welch's Method ---
nperseg = 1024  # Length of each segment (adjust based on your data)
noverlap = nperseg // 2  # Overlap between segments (50% overlap)
freqs_welch, psd_welch = welch(eeg_data, fs, nperseg=nperseg, noverlap=noverlap)

# --- Method 3: Multitaper Method ---
psd_mt, freqs_mt = psd_array_multitaper(eeg_data, sfreq=fs, adaptive=True, normalization='full', verbose=False)

# --- Plot the PSD Comparisons ---
plt.figure(figsize=(12, 6))
plt.semilogy(freqs_periodogram, psd_periodogram, label="Raw Periodogram", alpha=0.8)
plt.semilogy(freqs_welch, psd_welch, label="Welch's Method", alpha=0.8)
plt.semilogy(freqs_mt, psd_mt, label="Multitaper Method", alpha=0.8)

plt.ylim([1e-6, None])

# Plot formatting
plt.title("Comparison of PSD Estimation Methods", fontsize=14)
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("PSD (Power/Frequency)", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(fontsize=12)
plt.show()