"""
thesis_exported_tracking_sandbox.py
===================================

Simple thesis-plot sandbox for NPZ files exported from visuals/VitalExtraction_plot.py.
This does NOT process raw radar data. It only loads already-exported tracking/signals.

Edit the CONFIG section, then use:

    S["track_id"]["raw"]["signal_name"]
    S["track_id"]["line"]["Line_0"]
    S["track_id"]["line"]["some_other_line"]

Run:
    python thesis_exported_tracking_sandbox.py
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
# CONFIG: this is the part you should edit
# =============================================================================

INPUT_NPZ = "exportedTrackingData/21_michal_auto_radar_horizontal_standing_vitals_export.npz"

# Crop after loading. Use None to keep everything, or e.g. (0, 3500).
FRAME_LIMIT: Optional[tuple[int, int]] = (0, 3500)

# Pick tracks manually. Use None to load every exported track.
# Example: SELECT_TRACK_IDS = ["3", "7"]
SELECT_TRACK_IDS: Optional[list[str]] = None

# Main track used by example plots. If None, first available track is used.
MAIN_TRACK_ID: Optional[str] = None

# Names can differ depending on export. The script prints available names at startup.
# Line_0 is treated by your old script as unwrapped phase.
PHASE_SOURCE_KIND = "line"      # "line" or "raw"
PHASE_SOURCE_NAME = "Line_0"    # often "Line_0" = unwrapped phase

FS_HZ = 20.0                    # exported data is usually frame_index2time=0.05 => 20 Hz
DETRENDED_HIGHPASS_HZ = 0.1
BR_BAND_HZ = (0.2, 0.8)
HR_BAND_HZ = (0.8, 3.0)
FILTER_ORDER = 4

# For thesis plots, filtfilt is usually the right answer: no phase delay.
# If you need causal debugging, change to False.
USE_ZERO_PHASE_FILTERING = True

SAVE_FIGURES = False
OUTPUT_DIR = "thesis_figures"


# =============================================================================
# Loading helpers
# =============================================================================

def _slice(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if FRAME_LIMIT is None:
        return arr
    a, b = FRAME_LIMIT
    return arr[a:b]


def load_export(path: str):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return data, metadata


def collect_frames(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    frames = {}
    for key in data.files:
        if key.startswith("tracks/") and key.endswith("/frames"):
            _, track_id, _ = key.split("/", 2)
            if SELECT_TRACK_IDS is None or track_id in SELECT_TRACK_IDS:
                frames[track_id] = _slice(np.asarray(data[key]))
    return frames


def collect_groups(data: np.lib.npyio.NpzFile, prefix: str) -> dict[str, dict[str, np.ndarray]]:
    """Return nested dict: signal_name -> track_id -> values."""
    groups = defaultdict(dict)
    for key in data.files:
        if not key.startswith(prefix + "/"):
            continue
        parts = key.split("/")
        if len(parts) < 3:
            continue
        _, track_id, signal_name = parts[:3]
        if SELECT_TRACK_IDS is not None and track_id not in SELECT_TRACK_IDS:
            continue
        groups[signal_name][track_id] = _slice(np.asarray(data[key]))
    return dict(groups)


def invert_groups(groups: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """signal -> track -> values  becomes  track -> signal -> values."""
    out = defaultdict(dict)
    for signal_name, by_track in groups.items():
        for track_id, values in by_track.items():
            out[track_id][signal_name] = values
    return dict(out)


def build_signal_bank(path: str):
    data, metadata = load_export(path)
    frames = collect_frames(data)
    raw_by_signal = collect_groups(data, "tracks")
    line_by_signal = collect_groups(data, "lines")

    raw_by_track = invert_groups(raw_by_signal)
    line_by_track = invert_groups(line_by_signal)

    track_ids = sorted(set(frames) | set(raw_by_track) | set(line_by_track), key=lambda x: int(x) if x.isdigit() else x)
    S = {}
    for tid in track_ids:
        S[tid] = {
            "frames": frames.get(tid, np.arange(_guess_len(raw_by_track.get(tid, {}), line_by_track.get(tid, {})))),
            "raw": raw_by_track.get(tid, {}),
            "line": line_by_track.get(tid, {}),
        }

    add_derived_signals(S)
    return S, metadata, raw_by_signal, line_by_signal


def _guess_len(*dicts: dict[str, np.ndarray]) -> int:
    for d in dicts:
        for v in d.values():
            return len(v)
    return 0


# =============================================================================
# Signal processing helpers for exported arrays
# =============================================================================

def apply_filter(x: np.ndarray, kind: str, cutoff, fs: float = FS_HZ, order: int = FILTER_ORDER) -> np.ndarray:
    b, a = sig.butter(order, cutoff, btype=kind, fs=fs)
    if USE_ZERO_PHASE_FILTERING:
        return sig.filtfilt(b, a, np.asarray(x, dtype=float))
    return sig.lfilter(b, a, np.asarray(x, dtype=float))


def analytic_features(x: np.ndarray, fs: float = FS_HZ) -> dict[str, np.ndarray]:
    z = sig.hilbert(np.asarray(x, dtype=float))
    phase = np.unwrap(np.angle(z))
    freq = np.r_[np.nan, np.diff(phase) * fs / (2 * np.pi)]
    return {
        "analytic": z,
        "inst_amp": np.abs(z),
        "inst_phase": phase,
        "inst_freq": freq,
    }


def safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.asarray(a, dtype=float) / (np.asarray(b, dtype=float) + eps)


def norm(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = np.nanstd(y)
    if s == 0 or not np.isfinite(s):
        return y - np.nanmean(y)
    return (y - np.nanmean(y)) / s


def get_signal(S: dict, track_id: str, kind: str, name: str) -> np.ndarray:
    """Convenience accessor: get_signal(S, "3", "line", "Line_0")."""
    return S[str(track_id)][kind][name]


def add_derived_signals(S: dict):
    """
    Adds derived signals under S[track_id]["derived"].
    Source is configured by PHASE_SOURCE_KIND and PHASE_SOURCE_NAME.
    """
    for track_id, T in S.items():
        T["derived"] = {}
        source_group = T.get(PHASE_SOURCE_KIND, {})
        if PHASE_SOURCE_NAME not in source_group:
            continue

        phase = np.asarray(source_group[PHASE_SOURCE_NAME], dtype=float)
        # If Line_0 is already unwrapped phase, unwrap does no harm except tiny discontinuity cleanup.
        unwrapped = np.unwrap(phase)
        detrended = apply_filter(unwrapped, "highpass", DETRENDED_HIGHPASS_HZ)
        br = apply_filter(unwrapped, "bandpass", BR_BAND_HZ)
        hr = apply_filter(unwrapped, "bandpass", HR_BAND_HZ)

        D = T["derived"]
        D["phase_source"] = phase
        D["unwrapped"] = unwrapped
        D["detrended"] = detrended
        D["br"] = br
        D["hr"] = hr
        D.update(analytic_features(detrended))
        D.update({f"br_{k}": v for k, v in analytic_features(br).items()})
        D.update({f"hr_{k}": v for k, v in analytic_features(hr).items()})


def track_time_or_frames(S: dict, track_id: str, use_seconds: bool = False) -> np.ndarray:
    frames = np.asarray(S[str(track_id)]["frames"])
    if use_seconds:
        return (frames - frames[0]) / FS_HZ
    return frames


# =============================================================================
# Plot style + examples to copy/paste
# =============================================================================

def thesis_style():
    plt.rcParams.update({
        "figure.figsize": (7.2, 5.0),
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.linestyle": "--",  
        "grid.alpha": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.frameon": True,          # was False — need frame for background
        "legend.facecolor": "white",     # white background
        "legend.edgecolor": "gray",      # optional border
        "legend.framealpha": 1.0,        # fully opaque
        "lines.linewidth": 1.25,
    })


def savefig(fig, name: str):
    if not SAVE_FIGURES:
        return
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")


def example_detrended_amp_and_phase(S: dict, track_id: str):
    """Top: detrended phase + instantaneous amplitude. Bottom: instantaneous phase."""
    x = track_time_or_frames(S, track_id, use_seconds=False)
    D = S[track_id]["derived"]

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5.0))

    ax0b = ax[0].twinx()
    ax[0].plot(x, D["detrended"], color="C0", label="detrended phase")
    ax0b.plot(x, D["inst_amp"], color="C1", alpha=0.85, label="instantaneous amplitude")
    ax[0].set_ylabel("phase [rad]")
    ax0b.set_ylabel("amplitude [rad]")
    ax[0].set_title(f"Track ID {track_id}")

    l1, lab1 = ax[0].get_legend_handles_labels()
    l2, lab2 = ax0b.get_legend_handles_labels()
    ax[0].legend(l1 + l2, lab1 + lab2, loc="upper right")

    ax[1].plot(x, D["inst_phase"], color="C2", label="instantaneous phase")
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("phase [rad]")
    ax[1].legend(loc="upper right")

    fig.tight_layout()
    savefig(fig, f"track_{track_id}_detrended_amp_phase")
    return fig, ax


def example_ratios(S: dict, track_id: str):
    """Ratio examples. Replace with your actual thesis ratio and keep the scaffolding."""
    x = track_time_or_frames(S, track_id, use_seconds=False)
    D = S[track_id]["derived"]

    hr_over_br_amp = safe_ratio(D["hr_inst_amp"], D["br_inst_amp"])
    hr_over_total_amp = safe_ratio(D["hr_inst_amp"], D["inst_amp"])

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))
    ax.plot(x, hr_over_br_amp, label="HR amp / BR amp")
    ax.plot(x, hr_over_total_amp, label="HR amp / detrended amp")
    ax.set_xlabel("frame")
    ax.set_ylabel("ratio [-]")
    ax.set_title(f"Amplitude ratios, track ID {track_id}")
    ax.legend()
    fig.tight_layout()
    savefig(fig, f"track_{track_id}_ratios")
    return fig, ax


def example_compare_tracks(S: dict, signal_kind: str = "derived", signal_name: str = "detrended"):
    """Overlay one signal across tracks, normalized so scale differences do not wreck the figure."""
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))
    for track_id, T in S.items():
        if signal_name not in T.get(signal_kind, {}):
            continue
        x = track_time_or_frames(S, track_id, use_seconds=False)
        y = T[signal_kind][signal_name]
        ax.plot(x, norm(y), label=f"ID {track_id}")
    ax.set_xlabel("frame")
    ax.set_ylabel("normalized value [-]")
    ax.set_title(f"{signal_kind}:{signal_name} across tracks")
    ax.legend()
    fig.tight_layout()
    savefig(fig, f"compare_tracks_{signal_kind}_{signal_name}")
    return fig, ax


def example_plot_any_exported_signal(S: dict, track_id: str, kind: str, name: str):
    """For raw/line signals exactly as exported, no extra magic."""
    x = track_time_or_frames(S, track_id, use_seconds=False)
    y = S[track_id][kind][name]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))
    ax.plot(x, y, label=f"{kind}:{name}")
    ax.set_xlabel("frame")
    ax.set_ylabel(name)
    ax.set_title(f"Track ID {track_id}: {kind}:{name}")
    ax.legend()
    fig.tight_layout()
    savefig(fig, f"track_{track_id}_{kind}_{name}")
    return fig, ax

def phase_rad2mm(sig_rad):
    lambda_mm = 5
    sig_mm = lambda_mm  * sig_rad/(4*np.pi)
    return sig_mm

def print_available(S: dict, metadata: dict):
    print("Loaded export")
    print("Data nickname:", metadata.get("data_nickname"))
    print("Selected track IDs in metadata:", metadata.get("selected_track_ids"))
    print("\nAvailable tracks:", list(S.keys()))
    for tid, T in S.items():
        print(f"\nTrack {tid}")
        print("  raw:    ", sorted(T["raw"].keys()))
        print("  line:   ", sorted(T["line"].keys()))
        print("  derived:", sorted(T["derived"].keys()))


if __name__ == "__main__":
    thesis_style()
    S, metadata, raw_by_signal, line_by_signal = build_signal_bank(INPUT_NPZ)
    print_available(S, metadata)

    if not S:
        raise RuntimeError("No tracks loaded. Check INPUT_NPZ / SELECT_TRACK_IDS.")

    track_id = MAIN_TRACK_ID or next(iter(S.keys()))
    if not S[track_id]["derived"]:
        raise RuntimeError(
            f"No derived signals for track {track_id}. "
            f"Check PHASE_SOURCE_KIND={PHASE_SOURCE_KIND!r}, PHASE_SOURCE_NAME={PHASE_SOURCE_NAME!r}. "
            "The script printed available raw/line names above."
        )

    # -------------------------------------------------------------------------
    # Copy/paste examples. Comment out what you do not need.
    # -------------------------------------------------------------------------
    # example_detrended_amp_and_phase(S, track_id)
    # example_ratios(S, track_id)
    # example_compare_tracks(S, signal_kind="derived", signal_name="detrended")

    # Example: plot exactly exported signal, once you know its name from the printout.
    # example_plot_any_exported_signal(S, track_id, kind="line", name="Line_0")
    # example_plot_any_exported_signal(S, track_id, kind="raw", name="some_raw_signal_name")

    # Your scratchpad:
    # D = S[track_id]["derived"]
    # frames = track_time_or_frames(S, track_id)
    # my_ratio = safe_ratio(D["hr_inst_amp"], D["br_inst_amp"])
    # plt.figure(figsize=(7.2, 3.0))
    # plt.plot(frames, my_ratio)
    # plt.xlabel("frame")
    # plt.ylabel("my ratio [-]")
    # plt.tight_layout()



    # sig_A_detrend =  S["A"]["line"]["Line_2"]
    # sig_A_inst_f =  S["A"]["line"]["Line_4"]
    # sig_A_inst_A =  S["A"]["line"]["Line_5"]
    # t_A = track_time_or_frames(S,"A",use_seconds=True)

    # sig_A_inst_A = phase_rad2mm(sig_A_inst_A)
    # sig_A_detrend = phase_rad2mm(sig_A_detrend)

    # fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.2))
    # ax[0].plot(t_A, sig_A_detrend, label="detrended")
    # ax[0].plot(t_A, sig_A_inst_A, label="inst. amplitude")
    # ax[0].set_xlabel("time [s]")
    # ax[0].set_ylabel("displacement [mm]")
    # # ax[0].set_title(f"Instan")
    # ax[0].legend()

    # ax[1].plot(t_A, sig_A_inst_f, label="inst. freq.")
    # # ax[1].plot(t_A, sig_A_inst_A, label="inst. freq")
    # ax[1].set_xlabel("time [s]")
    # ax[1].set_ylabel("frequency [Hz]")
    # # ax[1].set_title(f"Amplitude ratios: {track_name}")
    # ax[1].legend()
    # fig.tight_layout()



    sig_A_pwr = S["A"]["raw"]["power_all"]
    sig_RE_pwr = S["RE"]["raw"]["power_all"]
    sig_A_highD = S["A"]["raw"]["power_high_dop"]
    sig_RE_highD = S["RE"]["raw"]["power_high_dop"]
        
    sig_A_detrend =  S["A"]["line"]["Line_2"]
    sig_RE_detrend =  S["RE"]["line"]["Line_2"]
  
    t_A = track_time_or_frames(S,"A",use_seconds=True)
    t_RE = track_time_or_frames(S,"RE",use_seconds=True)

    sig_A_detrend = phase_rad2mm(sig_A_detrend)
    sig_RE_detrend = phase_rad2mm(sig_RE_detrend)

    fig, ax = plt.subplots(3, 1 ) #, figsize=(7.2, 3.2))
    ax[0].plot(t_A, sig_A_detrend, label="A")
    ax[0].plot(t_RE, sig_RE_detrend, label="RE")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("displacement [mm]")
    ax[0].set_title("Ph. unwrap detrended")
    ax[0].legend()

    

    ax[1].plot(t_A, sig_A_pwr, label="A")
    ax[1].plot(t_RE, sig_RE_pwr, label="RE")
    # ax[1].plot(t_A, sig_A_inst_A, label="inst. freq")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("amplitude [1]")
    ax[1].set_title("Reflected signal amplitude, all doppler  bins")
    ax[1].legend()
    ax[1].set_yscale('log')

    ax[2].plot(t_A, sig_A_highD, label="A")
    ax[2].plot(t_RE, sig_RE_highD, label="RE")
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("amplitude [1]")
    ax[2].set_title("Reflected signal amplitude, higher doppler bins")
    ax[2].legend()
    ax[2].set_yscale('log')

    for a in ax:
        a.legend().set_draggable(True)

    fig.tight_layout(h_pad=0.8)








    plt.show()
