import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


fs = 20
# Filter specs
fc_BR = (0.2, 0.8)   # Hz
order_BR = 4



# Design filter
b_BR, a_BR = scipy.signal.butter(
    order_BR,
    fc_BR,
    btype='bandpass',
    fs=fs
)

# Frequency response
w, h = scipy.signal.freqz(b_BR, a_BR, worN=4096, fs=fs)

# Magnitude and phase
mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
phase = np.unwrap(np.angle(h[mag_db>-100] ))

# Plot
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# ----- Magnitude -----
ax[0].plot(w, mag_db, linewidth=2)
ax[0].set_title("Bandpass Filter Frequency Response")
ax[0].set_ylabel("Magnitude (dB)")
ax[0].grid(True)

# Edge frequency markers
for fc in fc_BR:
    ax[0].axvline(fc, color='red', linestyle='--', linewidth=1.5)

# ----- Phase -----
ax[1].plot(w[mag_db>-100], phase, linewidth=2)
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (radians)")
ax[1].grid(True)

# Edge frequency markers
for fc in fc_BR:
    ax[1].axvline(fc, color='red', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.show()