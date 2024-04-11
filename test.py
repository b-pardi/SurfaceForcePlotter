from main import read_lvm, COLUMN_HEADERS, get_sr
import numpy as np
import matplotlib.pyplot as plt

data = read_lvm('data/2024-01-10-Test_2.lvm')
y_fft = np.fft.fft(data['Voltage_0'])
N = len(data['Voltage_0'])
T = data['X_Value'].iloc[1] - data['X_Value'].iloc[0]  # Sampling interval
freq = np.fft.fftfreq(N, T)[:N//2]

# Calculate the magnitude of the FFT
magnitude = np.abs(y_fft)[:N // 2] * 1 / N

# Plotting the Frequency Spectrum
plt.figure(figsize=(14, 6))
plt.plot(freq, magnitude, color='red')
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()