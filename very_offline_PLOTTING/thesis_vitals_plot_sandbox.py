"""
Thesis-friendly vital-sign plotting sandbox.

Goal: keep the heavy radar loading/processing hidden, then give you simple arrays:

    S["track_name"]["detrended"]
    S["track_name"]["inst_amp"]
    S["track_name"]["inst_phase"]
    S["track_name"]["inst_freq"]
    S["track_name"]["hr"]
    S["track_name"]["br"]

You should mostly edit only:
    1) FILE_PATH
    2) TRACKS
    3) FRAME_LIMIT
    4) the plotting section at the bottom

Run:
    python thesis_vitals_plot_sandbox.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

# Script lives in HR_explorer/very_offline_processing.
# Add HR_explorer root so FileIO and processing imports work.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process


# =============================================================================
# 1) EDIT ME: input file + processing parameters
# =============================================================================

FILE_PATH = "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_21_michalauto_radarvodorovne_stani_24-04-2026_14-32-50.npz"

PARAMS = {
    "DoA_azi_N_elements": 8,
    "DoA_ele_N_elements": 1,
    "DoA_azi_range_degs": 90,
    "DoA_ele_range_degs": 30,
    "Channel_processing": "None",   # "None" or "DoA_customFFT"
    "Doppler_processing": "FFT",    # "FFT" or "None"
    "range_index2dist": 0.046,
    "frame_index2time": 5e-2,       # 20 Hz
    "doppler_index2freq": 1 / (5.76e-3),
    "doppler_index2vel": 0.157,
    "radarRotation_deg": 0,
}

# Process only this block first. Later you can crop even harder with crop().
PROCESS_FRAME_BEGIN_OFFSET = 100
PROCESS_FRAME_END_OFFSET = -600
PROCESS_RANGE_BEGIN = 16
PROCESS_RANGE_END = 22

# Then optionally crop the final signals for the figure.
# Use None to keep all processed frames, or e.g. (300, 900).
FRAME_LIMIT: Optional[tuple[int, int]] = None

# Manual track selection. These are penteract indices:
# penteract[:, doppler, range_relative, ele, azi]
# range_abs is the original absolute range bin; range_relative is computed for you.
TRACKS = [
    {"name": "main", "doppler": 0, "range_abs": 19, "ele": 0, "azi": 3},
    # {"name": "nearby_range", "doppler": 0, "range_abs": 20, "ele": 0, "azi": 3},
    # {"name": "nearby_azi",   "doppler": 0, "range_abs": 19, "ele": 0, "azi": 4},
]

# Filters. Change these freely.
DETRENDED_HIGHPASS_HZ = 0.1
BR_BAND_HZ = (0.2, 0.8)
HR_BAND_HZ = (0.8, 3.0)
FILTER_ORDER = 4
USE_ZERO_PHASE_FILTERING = True  # True=filtfilt for plots; False=lfilter for causal-ish debugging.


# =============================================================================
# 2) Utilities: you probably do not need to touch these
# =============================================================================

@dataclass
class Track:
    name: str
    doppler: int
    range_abs: int
    ele: int
    azi: int


def apply_filter(x: np.ndarray, fs: float, kind: str, cutoff, order: int = FILTER_ORDER) -> np.ndarray:
    b, a = sig.butter(order, cutoff, btype=kind, fs=fs)
    if USE_ZERO_PHASE_FILTERING:
        return sig.filtfilt(b, a, x)
    return sig.lfilter(b, a, x)


def analytic_features(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Instantaneous amplitude, phase, and frequency from Hilbert analytic signal."""
    z = sig.hilbert(x)
    phase = np.unwrap(np.angle(z))
    freq = np.r_[np.nan, np.diff(phase) * fs / (2 * np.pi)]
    return {
        "analytic": z,
        "inst_amp": np.abs(z),
        "inst_phase": phase,
        "inst_freq": freq,
    }


def safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ratio that does not explode on exact zeros."""
    return np.asarray(a) / (np.asarray(b) + eps)


def crop(t: np.ndarray, y: np.ndarray, frames: Optional[tuple[int, int]] = FRAME_LIMIT):
    """Crop by frame/sample index inside already-processed data."""
    if frames is None:
        return t, y
    i0, i1 = frames
    return t[i0:i1], y[i0:i1]


def normalize_for_overlay(y: np.ndarray) -> np.ndarray:
    """For plotting signals with different units on the same axis."""
    y = np.asarray(y, dtype=float)
    std = np.nanstd(y)
    if std == 0 or not np.isfinite(std):
        return y - np.nanmean(y)
    return (y - np.nanmean(y)) / std


def load_signal_bank(file_path: str, params: dict, tracks: list[dict]) -> tuple[np.ndarray, dict, float, dict]:
    """
    Returns:
        t: time vector [s]
        S: dict of named tracks/signals
        fs: sampling frequency [Hz]
        params: final processing params
    """
    loaded = loadNPZ(file_path)
    frames = loaded["frames"]

    params = HR_process.defaultSliders(frames, params.copy())
    params["i_Frames_begin"] += PROCESS_FRAME_BEGIN_OFFSET
    params["i_Frames_end"] += PROCESS_FRAME_END_OFFSET
    params["i_Range_begin"] = PROCESS_RANGE_BEGIN
    params["i_Range_end"] = PROCESS_RANGE_END

    fs = 1 / params["frame_index2time"]
    t0 = params["i_Frames_begin"] * params["frame_index2time"]

    penteract, _ = HR_process.process_A(frames, params)
    n = penteract.shape[0]
    t = t0 + np.arange(n) / fs

    S = {}
    for tr_dict in tracks:
        tr = Track(**tr_dict)
        range_rel = tr.range_abs - params["i_Range_begin"]
        complex_signal = penteract[:, tr.doppler, range_rel, tr.ele, tr.azi]

        phase = np.angle(complex_signal)
        unwrapped = np.unwrap(phase)
        detrended = apply_filter(unwrapped, fs, "highpass", DETRENDED_HIGHPASS_HZ)
        br = apply_filter(unwrapped, fs, "bandpass", BR_BAND_HZ)
        hr = apply_filter(unwrapped, fs, "bandpass", HR_BAND_HZ)

        track_signals = {
            "complex": complex_signal,
            "magnitude": np.abs(complex_signal),
            "phase": phase,
            "unwrapped": unwrapped,
            "detrended": detrended,
            "br": br,
            "hr": hr,
            "meta": tr,
        }
        track_signals.update(analytic_features(detrended, fs))
        track_signals.update({f"br_{k}": v for k, v in analytic_features(br, fs).items()})
        track_signals.update({f"hr_{k}": v for k, v in analytic_features(hr, fs).items()})
        S[tr.name] = track_signals

    return t, S, fs, params


def thesis_style():
    """A decent Matplotlib baseline: not fancy, just thesis-safe."""
    plt.rcParams.update({
        "figure.figsize": (7.2, 5.0),
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
    })


# =============================================================================
# 3) Plot examples: copy/paste/edit this section like a sane person
# =============================================================================

def example_plot_detrended_amp_phase(t: np.ndarray, S: dict, track_name: str = "main"):
    """
    Example requested by you:
      top subplot: detrended phase + instantaneous amplitude
      bottom subplot: instantaneous phase
    """
    x = S[track_name]
    tt, detrended = crop(t, x["detrended"])
    _, inst_amp = crop(t, x["inst_amp"])
    _, inst_phase = crop(t, x["inst_phase"])

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5.0))

    ax0b = ax[0].twinx()
    ax[0].plot(tt, detrended, color="C0", label="detrended phase")
    ax0b.plot(tt, inst_amp, color="C1", label="instantaneous amplitude", alpha=0.85)
    ax[0].set_ylabel("phase [rad]")
    ax0b.set_ylabel("amplitude [rad]")
    ax[0].set_title(f"Track: {track_name}")

    lines_a, labels_a = ax[0].get_legend_handles_labels()
    lines_b, labels_b = ax0b.get_legend_handles_labels()
    ax[0].legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")

    ax[1].plot(tt, inst_phase, color="C2", label="instantaneous phase")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("phase [rad]")
    ax[1].legend(loc="upper right")

    fig.tight_layout()
    return fig, ax


def example_plot_ratios(t: np.ndarray, S: dict, track_name: str = "main"):
    """
    Example ratio plot. Replace these with whatever ratio your thesis actually needs.
    """
    x = S[track_name]

    hr_to_br_amp = safe_ratio(x["hr_inst_amp"], x["br_inst_amp"])
    hr_to_detrended_amp = safe_ratio(x["hr_inst_amp"], x["inst_amp"])

    tt, r1 = crop(t, hr_to_br_amp)
    _, r2 = crop(t, hr_to_detrended_amp)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))
    ax.plot(tt, r1, label="HR inst. amp / BR inst. amp")
    ax.plot(tt, r2, label="HR inst. amp / detrended inst. amp")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ratio [-]")
    ax.set_title(f"Amplitude ratios: {track_name}")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def example_plot_multiple_tracks(t: np.ndarray, S: dict, signal_name: str = "detrended"):
    """Overlay the same signal from all selected tracks."""
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))
    for name, x in S.items():
        tt, yy = crop(t, x[signal_name])
        ax.plot(tt, normalize_for_overlay(yy), label=f"{name}: {signal_name}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("normalized value [-]")
    ax.set_title(f"Manual track comparison: {signal_name}")
    ax.legend()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    thesis_style()
    t, S, fs, params = load_signal_bank(FILE_PATH, PARAMS, TRACKS)

    print("Available tracks:", list(S.keys()))
    print("Available signals per track:")
    print(sorted(k for k in S[list(S.keys())[0]].keys() if k != "meta"))
    print(f"fs = {fs:.3f} Hz")

    # ---- Choose the examples you want by commenting/uncommenting. ----
    example_plot_detrended_amp_phase(t, S, track_name="main")
    example_plot_ratios(t, S, track_name="main")
    # example_plot_multiple_tracks(t, S, signal_name="detrended")
    # example_plot_multiple_tracks(t, S, signal_name="hr_inst_amp")

    # ---- Your own scratchpad example ----
    # x = S["main"]
    # tt, y = crop(t, x["hr"])
    # plt.figure(figsize=(7.2, 3.0))
    # plt.plot(tt, y)
    # plt.xlabel("time [s]")
    # plt.ylabel("HR-band phase [rad]")
    # plt.tight_layout()

    plt.show()
