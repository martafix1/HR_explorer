
import numpy as np
import sys, os

import scipy.signal

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process

import matplotlib.pyplot as plt
import mplcursors

params = {  "DoA_azi_N_elements":8,
            "DoA_ele_N_elements":1,
            "DoA_azi_range_degs":90, "DoA_ele_range_degs":30,
            # "Channel_processing":"DoA_customFFT",
            "Channel_processing":"None",
            "Doppler_processing":"FFT",

            "range_index2dist"  : 0.046,
            "frame_index2time"  : 5e-2,
            "doppler_index2freq"  : 1/(5.76* 1e-3), # Active chirping time 5.76 ms
            "doppler_index2vel"  : 0.157, # Velocity resolution, m/s
            "radarRotation_deg" : 0
                   }




# filePath = "data/unR_meas_noHR_32_rdr227_humancenter_06-03-2026_14-30-14.npz" 
# filePath = "data/unR_meas_noHR_33_rdr227_humancenterlowsitting_06-03-2026_14-33-03.npz"

# filePath = "data/unR_meas_7_josef_motorbezi_08-08-2025_13-38-58.npz"
# filePath = "data/unR_meas_9_josef_jizda_08-08-2025_13-43-06.npz"

filePath = "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_21_michalauto_radarvodorovne_stani_24-04-2026_14-32-50.npz"

loadedData = loadNPZ(filePath)

frames = loadedData["frames"]

params = HR_process.defaultSliders(frames,params)




frames0 = 100
frames1 = -600

params["i_Frames_begin"] += frames0
params["i_Frames_end"] += frames1

params["i_Range_begin"] = 16
params["i_Range_end"] = 22



t0 = params["i_Frames_begin"] * params["frame_index2time"]
t1 = params["i_Frames_end"] * params["frame_index2time"]

t = np.linspace(t0,t1, params["i_Frames_end"] -params["i_Frames_begin"] )


doppler_bin = 0
ele_bin = 0
azi_bin = 3
range_bin = 19-  params["i_Range_begin"]


penteract, DoA_dict = HR_process.process_A(frames,params)

data_new = penteract[:,doppler_bin,range_bin,ele_bin,azi_bin]
phase = np.angle(data_new)
magnitude = np.abs(data_new)
phase_unwrp = np.unwrap(phase)


fs = 1/params["frame_index2time"]  # Hz sampling freq

## Show OG, and detrending
print("To plot detrending")
order_DeTren = 4
fc_DeTren = 0.1
b_DeTren, a_DeTren = scipy.signal.butter(order_DeTren, fc_DeTren, btype='highpass',fs = fs)
DeTren_signal = scipy.signal.filtfilt(b_DeTren, a_DeTren, phase_unwrp)

order_DeTren_02 = 4
# fc_DeTren_02 = 0.9
fc_DeTren_02 = 0.2
b_DeTren_02, a_DeTren_02 = scipy.signal.butter(order_DeTren, fc_DeTren_02, btype='highpass',fs = fs)
DeTren_signal_02 = scipy.signal.filtfilt(b_DeTren_02, a_DeTren_02, phase_unwrp)


plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax1.plot(t,phase_unwrp,label = "og signal")
ax1.plot(t,phase_unwrp-DeTren_signal,label = "trend < 0.1Hz")
ax1.plot(t,phase_unwrp-DeTren_signal_02,label = f"trend < {fc_DeTren_02} Hz")
ax1.set_xlabel("t [s]")
ax1.set_ylabel("phase [rad]")
ax1.set_title(f"Phase unwrapping for [{params["i_Frames_begin"]}:{params["i_Frames_end"]},{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- (i_Frames,i_Doppler,i_Range,i_Ele,i_Azi), DoA: {params["Channel_processing"]}, Doppler: {params["Doppler_processing"]} ")
ax1.legend()
ax1.grid(True)

plt.subplot(2, 1, 2)
ax2.plot(t,DeTren_signal,label = "detrended signal (0.1 Hz)")
ax2.plot(t,DeTren_signal_02,label = "detrended signal (0.2 Hz)")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)

# plt.show() # tmp


## Try autocorrelation (short sliding window autocorrelation)


# def my_sliding_window_autocorrelation(signal,sliding_win_range : tuple, step_size):
    
#     scipy.signal.correlate(segment, segment, mode='full')

#     pass



def sliding_autocorrelation(sig, t, window_size, step_size, max_lag=None):
    """
    Compute sliding-window autocorrelation.

    Parameters:
        sig : 1D numpy array
        t   : time array (same length as sig)
        window_size : number of samples in each window
        step_size   : step between windows (samples)
        max_lag     : max lag to keep (samples)

    Returns:
        ac_matrix : 2D array (window_index x lag)
        lags      : lag values (in seconds)
        t_centers : time centers of each window
    """
    n = len(sig)

    if max_lag is None:
        max_lag = window_size // 2

    ac_list = []
    t_centers = []

    for start in range(0, n - window_size, step_size):
        segment = sig[start:start + window_size]

        # Remove mean (important for autocorrelation)
        segment = segment - np.mean(segment)

        # Compute autocorrelation
        ac = scipy.signal.correlate(segment, segment, mode='full')

        ac = ac*ac
        # Keep only positive lags
        mid = len(ac) // 2
        ac = ac[mid:mid + max_lag]

        # Normalize
        ac = ac / np.max(ac)

        ac_list.append(ac)

        # Time center of window
        t_centers.append(t[start + window_size // 2])

    ac_matrix = np.array(ac_list)
    lags = np.arange(max_lag) * (t[1] - t[0])

    return ac_matrix, lags, np.array(t_centers)

def plot_autocorrelation(ac_matrix, lags, t_centers):
    plt.figure(figsize=(10, 6))

    # --- log transform of amplitude ---
    # add small epsilon to avoid log(0)
    eps = 1e-10
    ac_log = np.log10(np.clip(ac_matrix, eps, None))

    plt.pcolormesh(
        t_centers,
        lags,
        ac_log.T,
        shading='auto',
        cmap='viridis'
    )

    plt.xlabel("Time")
    plt.ylabel("Lag (samples)")
    plt.title("Sliding Window Autocorrelation (log amplitude)")

    cbar = plt.colorbar()
    cbar.set_label("log10(autocorrelation)")

    plt.tight_layout()
    plt.show()


# window_size = 200     # samples
# step_size = 10        # overlap between windows
# max_lag = window_size           # limit lag

# ac_matrix, lags, t_centers = sliding_autocorrelation(
#     DeTren_signal_02, t, window_size, step_size, max_lag
# )

# plot_autocorrelation(ac_matrix, lags, t_centers)

def plot_spectrum(signal, fs, detrend=False, window=False):
    """
    Plot amplitude and phase spectrum of a signal.

    Parameters:
    - signal : array-like
    - fs : sampling rate (Hz)
    - detrend : remove mean before FFT
    - window : apply Hann window
    """

    # Ensure numpy array
    x = np.asarray(signal)

    if detrend:
        x = x - np.mean(x)

    N = len(x)

    # Optional windowing (reduces spectral leakage)
    if window:
        w = np.hanning(N)
        x = x * w
    else:
        w = np.ones(N)

    # FFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Amplitude correction (account for window energy)
    scale = np.sum(w) / N
    amplitude = np.abs(X) / (N * scale)
    amplitude[1:-1] *= 2  # single-sided spectrum correction

    phase = np.angle(X)

   
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    

    line1, = ax1.plot(freqs, amplitude)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Amplitude Spectrum")
    ax1.grid(True)

    line2, = ax2.plot(freqs, phase)
    ax2.set_ylabel("Phase [rad]")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_title("Phase Spectrum")
    ax2.grid(True)

    cursor = mplcursors.cursor([line1, line2], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f"f = {x:.2f} Hz\nval = {y:.4f}")

    plt.tight_layout()
    plt.show()

plot_spectrum(DeTren_signal_02[400:600],fs=20)


fc_BR = (0.2,0.8)  # Hz, cutoff freq
order_BR = 4  # good default

b_BR, a_BR = scipy.signal.butter(order_BR, fc_BR, btype='bandpass',fs = fs)
BR_signal = scipy.signal.filtfilt(b_BR, a_BR, phase_unwrp)


fc_HR = (0.8,3)  # Hz, cutoff freq
# fc_HR = (1.4,9)  # Hz, cutoff freq tried higher freqs
order_HR = 4  # good default

b_HR, a_HR = scipy.signal.butter(order_HR, fc_HR, btype='bandpass',fs = fs)
HR_signal = scipy.signal.filtfilt(b_HR, a_HR, phase_unwrp)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t,BR_signal,label = "breath rate signal")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)
plt.title(f"Phase unwrapping for [{params["i_Frames_begin"]}:{params["i_Frames_end"]},{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- (i_Frames,i_Doppler,i_Range,i_Ele,i_Azi), DoA: {params["Channel_processing"]}, Doppler: {params["Doppler_processing"]} ")

plt.subplot(2, 1, 2)
plt.plot(t,HR_signal,label = "heart rate signal")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)


plt.tight_layout()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t,BR_signal,label = "breath rate signal")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)
plt.title(f"Phase unwrapping for [{params["i_Frames_begin"]}:{params["i_Frames_end"]},{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- (i_Frames,i_Doppler,i_Range,i_Ele,i_Azi), DoA: {params["Channel_processing"]}, Doppler: {params["Doppler_processing"]} ")


analytic_BR = scipy.signal.hilbert(BR_signal)
phase_analytic_BR = np.unwrap(np.angle(analytic_BR))
inst_freq_BR = np.diff(phase_analytic_BR) * fs / (2 * np.pi)

plt.subplot(2, 1, 2)
plt.plot(t[:-1],inst_freq_BR,label = "BR freq")
plt.xlabel("t [s]")
plt.ylabel("freq [Hz]")
plt.legend()
plt.grid(True)


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t,HR_signal,label = "heart rate signal")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)
plt.title(f"Phase unwrapping for [{params["i_Frames_begin"]}:{params["i_Frames_end"]},{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- (i_Frames,i_Doppler,i_Range,i_Ele,i_Azi), DoA: {params["Channel_processing"]}, Doppler: {params["Doppler_processing"]} ")


analytic_HR = scipy.signal.hilbert(HR_signal)
phase_analytic_HR = np.unwrap(np.angle(analytic_HR))
inst_freq_HR = np.diff(phase_analytic_HR) * fs / (2 * np.pi)

plt.subplot(2, 1, 2)
plt.plot(t[:-1],inst_freq_HR,label = "HR freq")
plt.xlabel("t [s]")
plt.ylabel("freq [Hz]")
plt.legend()
plt.grid(True)


plt.show()