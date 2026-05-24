import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process


# -----------------------
# Hardcoded config section
# -----------------------
FILE_PATH = "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_21_michalauto_radarvodorovne_stani_24-04-2026_14-32-50.npz"

PARAMS = {
    "DoA_azi_N_elements": 8,
    "DoA_ele_N_elements": 1,
    "DoA_azi_range_degs": 90,
    "DoA_ele_range_degs": 30,
    "Channel_processing": "None",
    "Doppler_processing": "FFT",
    "range_index2dist": 0.046,
    "frame_index2time": 5e-2,
    "doppler_index2freq": 1 / (5.76e-3),
    "doppler_index2vel": 0.157,
    "radarRotation_deg": 0,
}

FRAMES_BEGIN_OFFSET = 100
FRAMES_END_OFFSET = -600
RANGE_BEGIN = 16
RANGE_END = 22

DOPPLER_BIN = 0
ELE_BIN = 0
AZI_BIN = 3
RANGE_BIN_ABSOLUTE = 19

DETRENDED_HIGHPASS_HZ = 0.1
DETRENDED_ORDER = 4

# One sample can be cut directly from the loaded/detrended signal.
USE_SIGNAL_SLICE_SAMPLE = True
SIGNAL_SAMPLE_SLICE = slice(200, 260)

# Literal samples: just edit these values. Keep them short or long; correlate() handles both.
fs = 20
sinsig = np.sin( np.arange(0,20)/fs * 2*3.14 )

HARDCODED_SAMPLES = [
    np.array([0.00, 0, 1, 1]),
    np.array([0.00, 0, 1, 1, 1, 0, 0]),
    np.array([0.00, 0, 1, 1, 1, 1, 1, 1, 0, 0]),
    np.array([0.00, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    np.array([0.00, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    # np.array([0.00, -0.05, -0.14, -0.22, -0.27, -0.21, -0.10, 0.02, 0.12, 0.20, 0.24, 0.17, 0.06]),
    sinsig
]


# -----------------------
# Processing helpers
# -----------------------
def load_unwrapped_phase(file_path, params):
    loaded_data = loadNPZ(file_path)
    frames = loaded_data["frames"]

    params = HR_process.defaultSliders(frames, params.copy())
    params["i_Frames_begin"] += FRAMES_BEGIN_OFFSET
    params["i_Frames_end"] += FRAMES_END_OFFSET
    params["i_Range_begin"] = RANGE_BEGIN
    params["i_Range_end"] = RANGE_END

    penteract, _ = HR_process.process_A(frames, params)

    range_bin_relative = RANGE_BIN_ABSOLUTE - params["i_Range_begin"]
    complex_signal = penteract[:, DOPPLER_BIN, range_bin_relative, ELE_BIN, AZI_BIN]
    phase_unwrapped = np.unwrap(np.angle(complex_signal))

    fs = 1 / params["frame_index2time"]
    t0 = params["i_Frames_begin"] * params["frame_index2time"]
    t = t0 + np.arange(len(phase_unwrapped)) / fs

    return t, phase_unwrapped, fs, params


def lfilter_detrend_highpass(signal, fs, cutoff_hz=0.1, order=4):
    b, a = scipy.signal.butter(order, cutoff_hz, btype="highpass", fs=fs)
    return scipy.signal.lfilter(b, a, signal)


def zscore(x):
    x = np.asarray(x, dtype=float)
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def normalized_correlation(signal, sample):
    sig = zscore(signal)
    samp = zscore(sample)
    corr = scipy.signal.correlate(sig, samp, mode="valid") / len(samp)
    return corr


# -----------------------
# Main
# -----------------------
def main():
    t, phase_unwrapped, fs, params = load_unwrapped_phase(FILE_PATH, PARAMS)
    detrended = lfilter_detrend_highpass(
        phase_unwrapped,
        fs,
        cutoff_hz=DETRENDED_HIGHPASS_HZ,
        order=DETRENDED_ORDER,
    )

    samples = []
    if USE_SIGNAL_SLICE_SAMPLE:
        samples.append((f"signal slice {SIGNAL_SAMPLE_SLICE.start}:{SIGNAL_SAMPLE_SLICE.stop}", detrended[SIGNAL_SAMPLE_SLICE]))

    for i, sample in enumerate(HARDCODED_SAMPLES, start=1):
        samples.append((f"sample {i}", sample))

    correlations = []
    for name, sample in samples:
        corr = normalized_correlation(detrended, sample)
        corr_t = t[: len(corr)] + (len(sample) / 2) / fs
        correlations.append((name, corr_t, corr))

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)

    title = (
        f"Unwrapped phase/correlation: frames "
        f"{params['i_Frames_begin']}:{params['i_Frames_end']}, "
        f"bin [doppler={DOPPLER_BIN}, range={RANGE_BIN_ABSOLUTE}, ele={ELE_BIN}, azi={AZI_BIN}]"
    )
    fig.suptitle(title)
    
    axes[0].sharex(axes[2])

    axes[0].plot(t, phase_unwrapped, label="unwrapped phase ", alpha=0.75)
    axes[0].plot(t, detrended, label=f"highpass detrended > {DETRENDED_HIGHPASS_HZ} Hz", linewidth=1.0)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("phase [rad]")
    axes[0].grid(True)
    axes[0].legend()

    for name, sample in samples:
        sample_t = np.arange(len(sample)) / fs
        axes[1].plot(sample_t, zscore(sample), marker="o", label=name)
    axes[1].set_xlabel("sample time [s]")
    axes[1].set_ylabel("phase centered [rad]")
    axes[1].grid(True)
    axes[1].legend()

    for name, corr_t, corr in correlations:
        axes[2].plot(corr_t, corr, label=f"corr vs {name}")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("normalized correlation")
    axes[2].grid(True)
    axes[2].legend()

    

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
