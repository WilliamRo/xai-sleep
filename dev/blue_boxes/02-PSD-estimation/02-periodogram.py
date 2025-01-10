import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic signal (replace with your data)
fs = 256  # Sampling frequency in Hz
duration = 10  # Duration in seconds
frequencies = [10, 20]  # Frequencies of interest in Hz
t = np.arange(0, duration, 1/fs)  # Time vector

# Simulate a signal: sum of two sinusoids + Gaussian noise
signal = 0.5 * np.sin(2 * np.pi * frequencies[0] * t) + \
         0.3 * np.sin(2 * np.pi * frequencies[1] * t) + \
         0.2 * np.random.randn(len(t))

# Raw Periodogram Computation
n = len(signal)  # Number of samples
freqs = np.fft.rfftfreq(n, d=1/fs)  # Frequency bins (only positive frequencies)
fft_vals = np.fft.rfft(signal)  # Compute FFT (only up to Nyquist frequency)
psd = (1 / (fs * n)) * np.abs(fft_vals) ** 2  # Normalize and compute power

# Plot the Raw Periodogram
plt.figure(figsize=(10, 6))
plt.semilogy(freqs, psd, label="Raw Periodogram")
plt.title("Power Spectral Density (Raw Periodogram)", fontsize=14)
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("PSD (Power/Frequency)", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(fontsize=12)
plt.show()