import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import scipy


def sliding_rife_estimation(r, fs, window_sec=10.0, min_freq=0.1, max_freq=0.6):
    """
    Estimates the instantaneous breathing frequency using a sliding-window Rife algorithm.
    
    Parameters:
    - r: 1D numpy array, the raw radar signal.
    - fs: float, sampling frequency (e.g., 20.0 Hz).
    - window_sec: float, length of the sliding window in seconds.
    - min_freq, max_freq: floats, bounding box for expected breathing rates.
    
    Returns:
    - f_est_array: 1D numpy array of instantaneous frequencies, same length as r.
    """
    N_total = len(r)
    W = int(window_sec * fs)
    f_est_array = np.zeros(N_total)
    
    if W >= N_total:
        W = N_total // 2 # Fallback if signal is shorter than window
        
    half_W = W // 2
    
    for i in range(N_total):
        # 1. Extract local window (clamp to edges to avoid zero-padding artifacts)
        start = max(0, i - half_W)
        end = start + W
        if end > N_total:
            end = N_total
            start = end - W
            
        segment = r[start:end]
        
        # Mean-center to remove DC bias
        segment = segment - np.mean(segment)
        
        # 2. Compute FFT
        yf = np.abs(np.fft.fft(segment))
        
        # 3. Restrict search to the physiological breathing range
        min_bin = max(1, int(min_freq * W / fs))
        max_bin = int(max_freq * W / fs)
        
        search_range = yf[min_bin:max_bin]
        if len(search_range) == 0:
            f_est_array[i] = 0.3 # Fallback if something goes wrong
            continue
            
        # Find the max bin in our restricted range
        local_k = np.argmax(search_range)
        k = local_k + min_bin
        
        # 4. Rife Interpolation
        # Check which neighbor is higher to determine interpolation direction
        if k < W//2 - 1 and yf[k+1] > yf[k-1]:
            p = 1
        else:
            p = -1
            
        # Standard Rife fractional correction formula (for rectangular window)
        # delta = X[k+p] / (X[k] + X[k+p])
        delta = yf[k+p] / (yf[k] + yf[k+p])
        
        # Interpolated bin
        k_interp = k + p * delta
        
        print(f"{i/fs :.2f} [s]: f_b {k_interp :.3f} [Hz]")

        # Convert interpolated bin back to Hz
        f_est_array[i] = k_interp * fs / W
        
    # Optional: Apply a light moving average to smooth quantization jitter
    smooth_w = int(1.0 * fs) # 1 second smoothing
    f_est_array = np.convolve(f_est_array, np.ones(smooth_w)/smooth_w, mode='same')
        
    return f_est_array

def ramped_square_wave(period, ramp_duration, fs, amplitude=1.0, cycles=1):
    """
    Generate a square wave with linear ramps (trapezoidal wave).

    Parameters:
        period (float): Total period of the waveform (seconds)
        ramp_duration (float): Duration of each rising/falling edge (seconds)
        fs (float): Sampling rate (Hz)
        amplitude (float): Peak amplitude
        cycles (int): Number of periods to generate

    Returns:
        t (np.ndarray): Time vector
        y (np.ndarray): Signal amplitude vector
    """

    # Convert times to samples
    samples_per_period = int(period * fs)
    ramp_samples = int(ramp_duration * fs)

    if 2 * ramp_samples > samples_per_period:
        raise ValueError("Ramp durations are too long for the given period.")

    flat_samples = samples_per_period - 2 * ramp_samples

    # Build one period
    rise = np.linspace(0, amplitude, ramp_samples, endpoint=False)
    high = np.full(flat_samples // 2, amplitude)
    fall = np.linspace(amplitude, 0, ramp_samples, endpoint=False)
    low = np.zeros(flat_samples - len(high))

    one_period = np.concatenate([rise, high, fall, low])

    # Repeat for number of cycles
    y = np.tile(one_period, cycles)

    # Time vector
    t = np.arange(len(y)) / fs

    return t, y

def generate_test_data(fs, duration):
    """
    Generates synthetic radar vital sign data.
    - Breath rate: Frequency varying around 0.3 Hz
    - Harmonics: 2nd to 5th harmonics of the breath rate
    - Heart rate: Sine wave at 1.0 Hz
    """
    t = np.arange(0, duration, 1/fs)
    
    # 1. Frequency varying breath rate 
    # f_br_center = 0.3  # ceneter freqency
    # f_br_variation = 0.1 # PeakPeak instantaneous frequency wander
    # f_br_mod = 0.2 # Frequency of BR change, Hz
    # instantaneous_f_br = f_br_center + f_br_variation * np.sin(2 * np.pi * f_br_mod * t)
    
    # Trapeziod breath rate - this one is hardcore 
    t, trapeziod = ramped_square_wave(period=60,ramp_duration=20,fs=fs)
    instantaneous_f_br = 0.3 + trapeziod * 0.1

    # constant one
    # instantaneous_f_br = 0.3 + 0*t # needs the t to make it an array

    # Phase is the integral of instantaneous frequency
    phase_br = 2 * np.pi * np.cumsum(instantaneous_f_br) / fs
    
    resp_amplitude = 10
    # Base respiratory signal
    resp_signal = resp_amplitude * np.sin(phase_br)
    
    # 2. Harmonics (2nd to 5th)
    harmonics_signal = (
        0.1 * resp_amplitude * np.sin(2 * phase_br + 0.5) +
        0.3 * resp_amplitude * np.sin(3 * phase_br + 0) +
        0.1 * resp_amplitude * np.sin(4 * phase_br + 0.2) +
        0.1 * resp_amplitude * np.sin(5 * phase_br + 0.8)
    )
    
    # 3. Heart rate signal (centered around 1Hz)
    f_hr = 1.0
    hr_signal = 0.3 * np.sin(2 * np.pi * f_hr * t)
    
    # Total combined signal with a bit of random noise
    noise = 0.05 * np.random.randn(len(t))
    r = resp_signal + harmonics_signal + hr_signal + noise
    
    return t, r, hr_signal, instantaneous_f_br

def estimate_fundamental_frequency(r, fs):
    """Uses FFT to find the fundamental breathing frequency."""
    N = len(r)
    yf = np.abs(fft(r))
    xf = fftfreq(N, 1/fs)
    
    # Look for the strongest peak in the normal breathing range (0.1 - 0.5 Hz)
    valid_idx = np.where((xf > 0.1) & (xf < 0.5))[0]
    f_est = xf[valid_idx[np.argmax(yf[valid_idx])]]
    return f_est

def rlsrhs_filter(r, t, f_est_array, harmonics_to_suppress=[2, 3, 4, 5]):
    """
    Implements Algorithm 1: RLSRHS procedure.
    """
    N = len(r)
    num_harmonics = len(harmonics_to_suppress)
    
    # Number of weights: 2 for each harmonic (one for sine, one for cosine)
    M = 2 * num_harmonics
    
    # RLS Parameters (as defined in the article)
    delta = 0.01  # Small positive initialization constant
    lam = 0.99   # Forgetting factor (lambda) - allows tracking of varying frequency
    
    # Initialization
    W = np.zeros((M, 1))
    P = (1.0 / delta) * np.eye(M)
    
    e = np.zeros(N) # The output signal (harmonics suppressed)
    y = np.zeros(N) # The estimated harmonics
    
    phase_accum = 0.0  # ADD THIS BEFORE THE LOOP
    dt = 1.0 / fs      # Time step per sample

    for i in range(N):
        # Accumulate the phase based on the instantaneous frequency
        phase_accum += 2 * np.pi * f_est_array[i] * dt
        
        # Construct reference signal vector X(t)
        X = np.zeros((M, 1))
        idx = 0
        for k in harmonics_to_suppress:
            # Use the accumulated phase multiplied by the harmonic order
            X[idx, 0] = np.cos(k * phase_accum)
            X[idx+1, 0] = np.sin(k * phase_accum)
            idx += 2
            
        # Calculate filter output (the estimated harmonic interference)
        y[i] = np.dot(W.T, X)[0, 0]
        
        # Calculate error (which becomes our harmonic-suppressed signal)
        e[i] = r[i] - y[i]
        
        # Calculate gain vector K
        num = np.dot(P, X)
        den = lam + np.dot(X.T, num)[0, 0]
        K = num / den
        
        # Update filter coefficients W
        W = W + K * e[i]
        
        # Update inverse correlation matrix P
        P = (P - np.dot(K, np.dot(X.T, P))) / lam

    return e

# --- Main Execution ---

# 1. Setup and Data Generation
fs = 20.0  # 50ms = 20Hz sampling rate
duration = 60.0 # 60 seconds
t, r, ground_truth_hr, instantaneous_f_br = generate_test_data(fs, duration)

plt.figure()
plt.plot(t,r,label="total mixed signal")
plt.title("First 35s of original signal in time domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim(0,35)


# 2. Estimate Breathing Frequency
# f_est = estimate_fundamental_frequency(r, fs)
f_est_array = sliding_rife_estimation(r,fs,window_sec=10)
f_est_array_w20 = sliding_rife_estimation(r,fs,window_sec=20)
f_est_array_w7 = sliding_rife_estimation(r,fs,window_sec=7)

# hilbert to compare it
analytic_BR = scipy.signal.hilbert(r)
phase_analytic_BR = np.unwrap(np.angle(analytic_BR))
inst_freq_BR = np.diff(phase_analytic_BR) * fs / (2 * np.pi)

b, a = butter(4, 0.1, btype='lowpass', fs=fs)
inst_freq_BR_LP = filtfilt(b, a, inst_freq_BR)

# print(f"Estimated baseline breathing frequency: {f_est:.3f} Hz")

plt.figure()
plt.plot(t,instantaneous_f_br,label="reference freq")
plt.plot(t,f_est_array_w7,label="sliding & rife estimation, 7s win")
plt.plot(t,f_est_array,label="sliding & rife estimation, 10s win")
plt.plot(t,f_est_array_w20,label="sliding & rife estimation, 20s win")
plt.plot(t[:-1],inst_freq_BR_LP,label="Hilbert freq estimation, low pass")
plt.title("Estimated respiration freq in time ")
plt.xlabel("Time (s)")
plt.ylabel("Freq [Hz]")
plt.legend(loc='upper right')
plt.grid(True)



# 3. Apply RLSRHS algorithm
# The resulting signal 'e' still contains the main breathing frequency and the heart rate,
# but the respiratory harmonics that leak into the heart rate band are removed.
e = rlsrhs_filter(r, t, f_est_array, harmonics_to_suppress=[2, 3, 4, 5])

# 4. Extract the Heart Rate (Optional, to prove it worked)
# Now that harmonics are gone, a simple bandpass filter easily extracts the HR
b, a = butter(4, [0.7, 1.5], btype='bandpass', fs=fs)
hr_extracted_raw = filtfilt(b, a, r) # Filtering the raw signal (fails due to harmonics)
hr_extracted_clean = filtfilt(b, a, e) # Filtering the RLSRHS signal (succeeds)

# 5. Plotting the results
plt.figure(figsize=(12, 8))

# Plot Frequency Spectra
plt.subplot(2, 1, 1)
N = len(t)
xf = fftfreq(N, 1/fs)[:N//2]
yf_raw = np.abs(fft(r))[:N//2]
yf_clean = np.abs(fft(e))[:N//2]

plt.plot(xf, yf_raw, label="Raw Signal (with Harmonics)", alpha=0.7)
plt.plot(xf, yf_clean, label="After RLSRHS (Harmonics Suppressed)", alpha=0.7)
plt.xlim(0, 2.0)
plt.title("Frequency Spectrum: Raw vs RLSRHS Processed")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)

# Plot Time Domain Heart Rate Extraction
plt.subplot(2, 1, 2)
# Zooming in on a 10 second window to see the waveforms clearly
start_idx, end_idx = int(30*fs), int(40*fs)
plt.plot(t[start_idx:end_idx], ground_truth_hr[start_idx:end_idx], 
         label="Ground Truth HR", color='black', linestyle='--')
plt.plot(t[start_idx:end_idx], hr_extracted_raw[start_idx:end_idx], 
         label="Bandpass on Raw (Distorted by harmonics)", color='red', alpha=0.6)
plt.plot(t[start_idx:end_idx], hr_extracted_clean[start_idx:end_idx], 
         label="Bandpass on RLSRHS (Cleaned)", color='green')

plt.title("Heart Rate Extraction in Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()



# STFT section
plt.figure()



# Compute STFT
winLen_s = 10
f, t_stft, Zxx = scipy.signal.stft(r, fs=fs, nperseg= int(winLen_s*fs))

# Convert magnitude to dB
# Zxx_dB = 20 * np.log10(np.abs(Zxx) + 1e-10)

# Plot heatmap
plt.figure(figsize=(10, 6)) 
plt.pcolormesh(t_stft, f, np.abs(Zxx),  cmap='magma') #shading='gouraud',
plt.colorbar(label='Magnitude (dB)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('STFT - Original signal, winLen {winLen_s} s')
plt.tight_layout()

# Cut freqs for HR range
f_min = 0.8
f_max = 3
f_res = f[1]-f[0]
bin_fmin = int(f_min/f_res)
bin_fmax = int(f_max/f_res) 

# Plot heatmap
plt.figure(figsize=(10, 6)) 
plt.pcolormesh(t_stft, f[bin_fmin:bin_fmax], np.abs(Zxx[bin_fmin:bin_fmax,:]),  cmap='viridis') #shading='gouraud',
plt.colorbar(label='Magnitude (dB)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(f'STFT - OG, Zoom on HR freqs, winLen {winLen_s} s')
plt.tight_layout()


# STFT for E signal
f, t_stft, Zxx = scipy.signal.stft(e, fs=fs, nperseg= int(winLen_s*fs))

# Cut freqs for HR range
f_min = 0.8
f_max = 3
f_res = f[1]-f[0]
bin_fmin = int(f_min/f_res)
bin_fmax = int(f_max/f_res) 

# Plot heatmap
plt.figure(figsize=(10, 6)) 
plt.pcolormesh(t_stft, f[bin_fmin:bin_fmax], np.abs(Zxx[bin_fmin:bin_fmax,:]),  cmap='viridis') #shading='gouraud',
plt.colorbar(label='Magnitude (dB)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(f'STFT - Cleaned signal, Zoom on HR, winLen {winLen_s} s')
plt.tight_layout()


# Cut time to remove init noise
t_cut = 10
t_res = t_stft[1]-t_stft[0]
bit_tcut = int(t_cut/t_res)


# Plot heatmap
plt.figure(figsize=(10, 6)) 
plt.pcolormesh(t_stft[bit_tcut:-1], f[bin_fmin:bin_fmax], np.abs(Zxx[bin_fmin:bin_fmax,bit_tcut:-1]),  cmap='viridis') #shading='gouraud',
plt.colorbar(label='Magnitude (dB)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(f'STFT - Cleaned signal, Zoom on HR, Cut init, winLen {winLen_s} s')
plt.tight_layout()



plt.show()


