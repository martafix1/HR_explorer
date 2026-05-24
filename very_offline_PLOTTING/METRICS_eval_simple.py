"""
METRICS_eval_simple.py
======================

Half-hardcoded SNR plotting for exported tracking NPZ files.
No raw radar processing. Edit CONFIG, run, tweak plots.

Creates 3 figures by default:
    1) raw:  power_high_dop
    2) line: Line_2
    3) line: Line_5

Each figure has:
    top:    aligned/repaired signal + noise
    middle: linear SNR ratio abs(signal) / abs(noise)
    bottom: log SNR 20log10(abs(signal) / abs(noise))
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


# =============================================================================
# CONFIG - edit this part
# =============================================================================
FRAME_LIMIT = None

signals = ["A"] #,"B"
noise = "RK" #"RE" "RK"

# INPUT_NPZ = "exportedTrackingData/21_michal_auto_radar_horizontal_standing_vitals_export.npz";  signals = ["A","B"]; noise = "RE"
# INPUT_NPZ = "exportedTrackingData/22_michal_auto_radar_horizontal_driving_vitals_export__data120samples.npz" ; signals = ["A","B"]; noise = "R"  #,"B"
INPUT_NPZ = "exportedTrackingData/31_Kofola_big_drive_vitals_export_normalWindows.npz" # needs line 6 for BR f, 7 for BR a, 4,5 have too low of a f trehsold.
FRAME_LIMIT: Optional[tuple[int, int]] = (0, 9750) # friving only for 31
# INPUT_NPZ = "exportedTrackingData/28_Kofola_enter_repark_exit_vitals_export_BR_zeroPad400.npz"
# INPUT_NPZ = "exportedTrackingData/32_Kofola_parking_vitals_export_zeroPad400.npz"; FRAME_LIMIT: Optional[tuple[int, int]] = (0, 3000)

# 
# FRAME_LIMIT: Optional[tuple[int, int]] = (0, 3500)



# signals = ["A"] #,"B"
# noise = "RK" #"RE" "RK"

FS_HZ = 20.0
MAX_INTERP_GAP_FRAMES = 8

# Used only when convert_to_mm=True for phase lines.
# displacement_mm = phase_rad * LAMBDA_MM / (4*pi)
# Edit this if your radar wavelength is different.
LAMBDA_MM = 5.0

MEASURES = [
    {
        "kind": "raw",
        "name": "power_high_dop",
        "title": "High Doppler power",
        "ylabel": "amplitude [-]",
        "convert_to_mm": False,
        "lowpass_2s": True,
        "show_linear_snr": False,
        "log_y_signal": True,
    },
    {
        "kind": "line",
        "name": "Line_2",
        "title": "Phase unwrap detrended",
        "ylabel": "displacement [mm]",
        "convert_to_mm": True,
        "lowpass_2s": False,
        "show_linear_snr": False,
        "log_y_signal": False,
    },
    {
        "kind": "line",
        "name": "Line_5",
        "title": "BR instantaneous amplitude",
        "ylabel": "displacement [mm]",
        "convert_to_mm": True,
        "lowpass_2s": False,
        "show_linear_snr": True,
        "log_y_signal": False,
    },
    {
        "kind": "line",
        "name": "Line_4",
        "title": "BR Frequency",
        "ylabel": "frequency [Hz]",
        "convert_to_mm": False,
        "lowpass_2s": False,
        "show_linear_snr": False,
        "log_y_signal": False,
        "only_frequency_plot": True,
    }

]


# =============================================================================
# Small helpers
# =============================================================================

def phase_rad2mm(phase_rad):
    return np.asarray(phase_rad, dtype=float) * LAMBDA_MM / (4.0 * np.pi)


def simpleSNR_normal(noise, signal, eps=1e-12):
    return np.abs(signal) / (np.abs(noise) + eps)


def simpleSNR_20Log(noise, signal, eps=1e-12):
    return 20.0 * np.log10(simpleSNR_normal(noise, signal, eps=eps) + eps)


def lowpass_2s_lfilter(x, fs=FS_HZ, order=4):
    cutoff_hz = 1.0 / 2.0
    b, a = sig.butter(order, cutoff_hz, btype="lowpass", fs=fs)
    return sig.lfilter(b, a, np.asarray(x, dtype=float))


def stats_text(x):
    x = np.asarray(x, dtype=float)
    return f"mean={np.nanmean(x):.4g}, std={np.nanstd(x):.4g}"


# =============================================================================
# Function 1: load all exported signals into S
# =============================================================================

def load_export_signals(path, frame_limit=None, track_ids=None):
    """
    Returns:
        S[track_id]["frames"]
        S[track_id]["raw"][signal_name]
        S[track_id]["line"][line_name]
        metadata
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))

    if track_ids is not None:
        track_ids = {str(t) for t in track_ids}

    def keep_frame_mask(frames):
        if frame_limit is None:
            return np.ones(len(frames), dtype=bool)
        a, b = frame_limit
        return (frames >= a) & (frames <= b)

    S = defaultdict(lambda: {"frames": None, "raw": {}, "line": {}})

    for key in data.files:
        if key.startswith("tracks/") and key.endswith("/frames"):
            _, track_id, _ = key.split("/", 2)
            if track_ids is not None and track_id not in track_ids:
                continue
            frames = np.asarray(data[key], dtype=int)
            S[track_id]["frames"] = frames[keep_frame_mask(frames)]

    for key in data.files:
        if not key.startswith("tracks/"):
            continue
        parts = key.split("/")
        if len(parts) < 3:
            continue
        _, track_id, signal_name = parts[:3]
        if signal_name == "frames":
            continue
        if track_ids is not None and track_id not in track_ids:
            continue
        if S[track_id]["frames"] is None:
            continue

        original_frames = np.asarray(data[f"tracks/{track_id}/frames"], dtype=int)
        mask = keep_frame_mask(original_frames)
        S[track_id]["raw"][signal_name] = np.asarray(data[key])[mask]

    for key in data.files:
        if not key.startswith("lines/"):
            continue
        parts = key.split("/")
        if len(parts) < 3:
            continue
        _, track_id, line_name = parts[:3]
        if track_ids is not None and track_id not in track_ids:
            continue
        if S[track_id]["frames"] is None:
            continue

        original_frames = np.asarray(data[f"tracks/{track_id}/frames"], dtype=int)
        mask = keep_frame_mask(original_frames)
        S[track_id]["line"][line_name] = np.asarray(data[key])[mask]

    return dict(S), metadata


# =============================================================================
# Function 2: repair supports so selected signals can be compared
# =============================================================================

def repair_supports(S, selected, max_gap=MAX_INTERP_GAP_FRAMES, mode="intersection"):
    """
    Align selected signals onto a common frame support.

    Args:
        selected = [(track_id, kind, signal_name), ...]
        kind is "raw" or "line"

    Returns:
        frames_common, R

        R[(track_id, kind, signal_name)] = aligned_values
        R[track_id][kind][signal_name] = aligned_values
        R[track_id]["frames"] = frames_common

    This means both of these work:
        R[("A", "line", "Line_2")]
        R["A"]["line"]["Line_2"]
    """

    def regularize(frames, values):
        frames = np.asarray(frames, dtype=int)
        values = np.asarray(values, dtype=float)

        order = np.argsort(frames)
        frames = frames[order]
        values = values[order]

        _, unique_last_indices = np.unique(frames[::-1], return_index=True)
        keep = len(frames) - 1 - unique_last_indices
        keep = np.sort(keep)
        frames = frames[keep]
        values = values[keep]

        full_frames = np.arange(frames[0], frames[-1] + 1)
        full_values = np.full(len(full_frames), np.nan)
        full_values[frames - frames[0]] = values

        for i in range(len(frames) - 1):
            f0, f1 = frames[i], frames[i + 1]
            gap = f1 - f0 - 1
            if 0 < gap <= max_gap:
                idx0 = f0 - frames[0]
                idx1 = f1 - frames[0]
                full_values[idx0:idx1 + 1] = np.linspace(values[i], values[i + 1], gap + 2)

        return full_frames, full_values

    regularized = {}
    for track_id, kind, name in selected:
        track_id = str(track_id)
        frames = S[track_id]["frames"]
        values = S[track_id][kind][name]
        regularized[(track_id, kind, name)] = regularize(frames, values)

    if mode == "intersection":
        start = max(fr[0] for fr, _ in regularized.values())
        end = min(fr[-1] for fr, _ in regularized.values())
    elif mode == "union":
        start = min(fr[0] for fr, _ in regularized.values())
        end = max(fr[-1] for fr, _ in regularized.values())
    else:
        raise ValueError("mode must be 'intersection' or 'union'")

    frames_common = np.arange(start, end + 1)

    R = defaultdict(lambda: {"frames": frames_common, "raw": {}, "line": {}})
    flat_values = {}

    for key, (frames, values) in regularized.items():
        track_id, kind, name = key
        aligned = np.full(len(frames_common), np.nan)

        overlap_start = max(frames_common[0], frames[0])
        overlap_end = min(frames_common[-1], frames[-1])
        if overlap_end >= overlap_start:
            dst0 = overlap_start - frames_common[0]
            dst1 = overlap_end - frames_common[0] + 1
            src0 = overlap_start - frames[0]
            src1 = overlap_end - frames[0] + 1
            aligned[dst0:dst1] = values[src0:src1]

        flat_values[key] = aligned

    if mode == "intersection":
        valid = np.ones(len(frames_common), dtype=bool)
        for values in flat_values.values():
            valid &= np.isfinite(values)
        frames_common = frames_common[valid]
        flat_values = {key: values[valid] for key, values in flat_values.items()}

    for key, values in flat_values.items():
        track_id, kind, name = key
        R[track_id]["frames"] = frames_common
        R[track_id][kind][name] = values
        R[key] = values

    return frames_common, dict(R)


def track_time_or_frames(R: dict, track_id: str, use_seconds: bool = False):
    """Works with repaired R, not original S."""
    frames = np.asarray(R[str(track_id)]["frames"])
    if use_seconds:
        return (frames - frames[0]) / FS_HZ
    return frames


# =============================================================================
# Hardcoded plotting
# =============================================================================

def plot_frequency_measure(R, measure, freq_name=None):
    """
    Separate frequency plot for one measure.

    Y: Hz
    X: time [s]
    Plots both noise and all signal tracks.

    By default it uses the same kind as measure, and signal name:
        measure["freq_name"] if present
        otherwise freq_name argument
        otherwise measure["name"]

    Example:
        plot_frequency_measure(R, {
            "kind": "line",
            "name": "Line_5",
            "freq_name": "Line_6",
            "title": "Instantaneous frequency",
        })
    """
    kind = measure["kind"]
    name = freq_name or measure.get("freq_name", measure["name"])

    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5))
    fig.suptitle(measure.get("title", f"Frequency: {kind}:{name}"))
    
    # noise
    t_noise = track_time_or_frames(R, noise, use_seconds=True)
    y_noise = R[noise][kind][name]
    txt = stats_text(y_noise)
    ax.plot(t_noise, y_noise, label=f"noise {noise}: {txt}", color="k", alpha=0.75)

    # signals
    for sig_id in signals:
        t_sig = track_time_or_frames(R, sig_id, use_seconds=True)
        y_sig = R[sig_id][kind][name]
        txt = stats_text(y_sig)
        ax.plot(t_sig, y_sig, label=f"signal {sig_id}: {txt}")

        print(f"\nFrequency {kind}:{name}")
        print(f"  {sig_id}: {stats_text(y_sig)}")

    print(f"\nFrequency {kind}:{name} / noise {noise}")
    print(f"  {noise}: {stats_text(y_noise)}")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    # ax.set_title("frequency")
    ax.grid(True, alpha=0.3)

    leg = ax.legend(fontsize=8)
    if leg is not None:
        leg.set_draggable(True)

    fig.tight_layout()
    return fig, ax



def plot_measure(R, measure):
    kind = measure["kind"]
    name = measure["name"]
    show_linear_snr = measure.get("show_linear_snr", True)

    t_noise = track_time_or_frames(R, noise, use_seconds=True)
    y_noise = R[noise][kind][name]

    if measure["convert_to_mm"]:
        y_noise = phase_rad2mm(y_noise)

    if measure["lowpass_2s"]:
        y_noise = lowpass_2s_lfilter(y_noise)

    if show_linear_snr:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
        ax_signal = ax[0]
        ax_linear = ax[1]
        ax_log = ax[2]
    else:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        ax_signal = ax[0]
        ax_linear = None
        ax_log = ax[1]

    fig.suptitle(measure["title"])

    ax_signal.plot(t_noise, y_noise, label=f"noise {noise}", color="k", alpha=0.75)

    for sig_id in signals:
        t_sig = track_time_or_frames(R, sig_id, use_seconds=True)
        y_sig = R[sig_id][kind][name]

        if measure["convert_to_mm"]:
            y_sig = phase_rad2mm(y_sig)

        if measure["lowpass_2s"]:
            y_sig = lowpass_2s_lfilter(y_sig)

        snr_ratio = simpleSNR_normal(noise=y_noise, signal=y_sig)
        snr_log = simpleSNR_20Log(noise=y_noise, signal=y_sig)

        ax_signal.plot(t_sig, y_sig, label=f"signal {sig_id}")

        if show_linear_snr:
            ax_linear.plot(
                t_sig,
                snr_ratio,
                label=f"{sig_id}: {stats_text(snr_ratio)}",
            )

        ax_log.plot(
            t_sig,
            snr_log,
            label=f"{sig_id}: {stats_text(snr_log)}",
        )

        print(f"\n{measure['title']} / signal {sig_id}")
        print(f"  signal:    {stats_text(y_sig)}")
        print(f"  noise:     {stats_text(y_noise)}")

        if show_linear_snr:
            print(f"  SNR ratio: {stats_text(snr_ratio)}")

        print(f"  SNR log:   {stats_text(snr_log)} dB")

    ax_signal.set_ylabel(measure["ylabel"])
    if measure.get("lowpass_2s"):
        ax_signal.set_title("low passed signals")
    else:
        ax_signal.set_title("original signals")
    if measure.get("log_y_signal", False):
        ax_signal.set_yscale("log")

    if show_linear_snr:
        ax_linear.set_ylabel("ratio [-]")
        ax_linear.set_title("SNR linear")

    ax_log.set_xlabel("time [s]")
    ax_log.set_ylabel("log [dB]")
    ax_log.set_title("SNR log")
    ax_log.axhline(0, color="k", alpha=0.25, linewidth=0.8)

    for a in ax:
        a.grid(True, alpha=0.3)
        leg = a.legend(fontsize=8)
        if leg is not None:
            leg.set_draggable(True)

    fig.tight_layout(h_pad=0.8)
    return fig, ax



def main():
    S, meta = load_export_signals(
        INPUT_NPZ,
        frame_limit=FRAME_LIMIT,
        track_ids=[noise] + signals,
    )

    selected = []
    for measure in MEASURES:
        selected.append((noise, measure["kind"], measure["name"]))
        for sig_id in signals:
            selected.append((sig_id, measure["kind"], measure["name"]))

    frames, R = repair_supports(S, selected)

    print("Loaded tracks:", list(S.keys()))
    print("Common repaired support:", frames[0], "..", frames[-1], "N=", len(frames))

    for measure in MEASURES:
        if measure.get("only_frequency_plot", False):
            plot_frequency_measure(R, measure)
        else:
            plot_measure(R, measure)


    plt.show()


if __name__ == "__main__":
    main()
