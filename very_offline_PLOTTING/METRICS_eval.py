"""
METRICS_eval.py
===============

SNR / metrics sandbox for exported tracking NPZ files from VitalExtraction_plot.py.
No raw radar processing. This works on exported `tracks/...` and `lines/...` arrays.

What it does:
- chooses 1 noise track and 1+ signal tracks
- handles short missing-frame holes by interpolation
- aligns signal/noise tracks by frame number
- computes:
    1) power_high_dop amplitude, low-passed with Butterworth lfilter, cutoff period 2 s
       -> linear power ratio and log SNR [dB]
    2) phase-like signals
       -> phase difference [rad]
       -> absolute phase ratio
       -> log phase SNR [dB]
    3) mean/std for raw signals, line signals, and all SNR metrics
- plots everything

Edit only CONFIG first. Then run:
    python METRICS_eval.py
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
# CONFIG
# =============================================================================

INPUT_NPZ = "exportedTrackingData/21_michal_auto_radar_horizontal_standing_vitals_export.npz"

# Frame crop by absolute frame number. Use None to keep all.
FRAME_LIMIT: Optional[tuple[int, int]] = None  # e.g. (0, 3500)

# Track IDs are strings in the exported NPZ.
NOISE_TRACK_ID = "RE"
SIGNAL_TRACK_IDS = ["A"]  # e.g. ["1", "2", "5"]

# Raw amplitude-like signal. Script prints available names if this is wrong.
POWER_SIGNAL_KIND = "raw"  # usually "raw"
POWER_SIGNAL_NAME = "power_high_dop"

# Phase-like signals to evaluate. Line_0 was unwrapped phase in your old viewer.
PHASE_SIGNALS = [
    ("line", "Line_2"),
    ("line", "Line_5"),
    # ("raw", "some_phase_signal"),
]

FS_HZ = 20.0
LOWPASS_PERIOD_S = 2.0
LOWPASS_CUTOFF_HZ = 1.0 / LOWPASS_PERIOD_S
LOWPASS_ORDER = 4

# Missing support handling: gaps up to this many frames are linearly interpolated.
# Longer gaps stay NaN and are ignored in stats/SNR.
MAX_INTERP_GAP_FRAMES = 8

# Numerical safety.
EPS = 1e-12

SAVE_FIGURES = False
OUTPUT_DIR = "metrics_figures"


# =============================================================================
# Loading / exported-data access
# =============================================================================

def load_export(path: str):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return data, metadata


def collect_frames(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    frames = {}
    for key in data.files:
        if key.startswith("tracks/") and key.endswith("/frames"):
            _, track_id, _ = key.split("/", 2)
            frames[track_id] = np.asarray(data[key], dtype=int)
    return frames


def collect_groups(data: np.lib.npyio.NpzFile, prefix: str) -> dict[str, dict[str, np.ndarray]]:
    """Return signal_name -> track_id -> values."""
    groups = defaultdict(dict)
    for key in data.files:
        if not key.startswith(prefix + "/"):
            continue
        parts = key.split("/")
        if len(parts) < 3:
            continue
        _, track_id, signal_name = parts[:3]
        groups[signal_name][track_id] = np.asarray(data[key])
    return dict(groups)


def invert_groups(groups: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """signal -> track -> values  becomes  track -> signal -> values."""
    out = defaultdict(dict)
    for signal_name, by_track in groups.items():
        for track_id, values in by_track.items():
            out[track_id][signal_name] = values
    return dict(out)


def build_bank(path: str):
    data, metadata = load_export(path)
    frames = collect_frames(data)
    raw = invert_groups(collect_groups(data, "tracks"))
    line = invert_groups(collect_groups(data, "lines"))

    track_ids = sorted(set(frames) | set(raw) | set(line), key=lambda x: int(x) if x.isdigit() else x)
    S = {}
    for tid in track_ids:
        S[tid] = {
            "frames": crop_by_frame(frames.get(tid, np.arange(guess_len(raw.get(tid, {}), line.get(tid, {}))))),
            "raw": {},
            "line": {},
        }
        original_frames = frames.get(tid, np.arange(guess_len(raw.get(tid, {}), line.get(tid, {}))))
        keep = frame_mask(original_frames)
        for name, values in raw.get(tid, {}).items():
            S[tid]["raw"][name] = np.asarray(values)[keep]
        for name, values in line.get(tid, {}).items():
            S[tid]["line"][name] = np.asarray(values)[keep]
    return S, metadata


def guess_len(*dicts: dict[str, np.ndarray]) -> int:
    for d in dicts:
        for v in d.values():
            return len(v)
    return 0


def frame_mask(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if FRAME_LIMIT is None:
        return np.ones(len(frames), dtype=bool)
    a, b = FRAME_LIMIT
    return (frames >= a) & (frames <= b)


def crop_by_frame(frames: np.ndarray) -> np.ndarray:
    return np.asarray(frames)[frame_mask(frames)]


def get_signal(S: dict, track_id: str, kind: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return frames, values for S[track_id][kind][name]."""
    track_id = str(track_id)
    if track_id not in S:
        raise KeyError(f"Missing track {track_id}. Available: {list(S)}")
    if name not in S[track_id][kind]:
        available = sorted(S[track_id][kind].keys())
        raise KeyError(f"Missing {kind}:{name} for track {track_id}. Available {kind}: {available}")
    return np.asarray(S[track_id]["frames"], dtype=int), np.asarray(S[track_id][kind][name], dtype=float)


# =============================================================================
# Missing-frame support / alignment
# =============================================================================

def regularize_with_short_gap_fill(frames: np.ndarray, values: np.ndarray):
    """
    Put irregular track samples on a full integer-frame axis.
    Missing gaps <= MAX_INTERP_GAP_FRAMES are linearly interpolated.
    Longer gaps remain NaN.
    """
    frames = np.asarray(frames, dtype=int)
    values = np.asarray(values, dtype=float)
    order = np.argsort(frames)
    frames = frames[order]
    values = values[order]

    # Drop duplicate frames, keeping the last occurrence.
    uniq_frames, last_idx = np.unique(frames[::-1], return_index=True)
    keep_idx = len(frames) - 1 - last_idx
    order2 = np.argsort(uniq_frames)
    frames = uniq_frames[order2]
    values = values[keep_idx[order2]]

    full_frames = np.arange(frames[0], frames[-1] + 1)
    full_values = np.full(len(full_frames), np.nan, dtype=float)
    full_values[frames - frames[0]] = values

    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        gap = f1 - f0 - 1
        if 0 < gap <= MAX_INTERP_GAP_FRAMES:
            idx0 = f0 - frames[0]
            idx1 = f1 - frames[0]
            full_values[idx0:idx1 + 1] = np.linspace(values[i], values[i + 1], gap + 2)

    return full_frames, full_values


def align_pair(fr_a: np.ndarray, a: np.ndarray, fr_b: np.ndarray, b: np.ndarray):
    """Regularize both tracks, then return common finite frame support."""
    fa, aa = regularize_with_short_gap_fill(fr_a, a)
    fb, bb = regularize_with_short_gap_fill(fr_b, b)

    start = max(fa[0], fb[0])
    end = min(fa[-1], fb[-1])
    if end < start:
        return np.array([], dtype=int), np.array([]), np.array([])

    ia0 = start - fa[0]
    ib0 = start - fb[0]
    n = end - start + 1
    frames = np.arange(start, end + 1)
    aa = aa[ia0:ia0 + n]
    bb = bb[ib0:ib0 + n]
    good = np.isfinite(aa) & np.isfinite(bb)
    return frames[good], aa[good], bb[good]


# =============================================================================
# Metrics
# =============================================================================

def lowpass_lfilter(x: np.ndarray) -> np.ndarray:
    b, a = sig.butter(LOWPASS_ORDER, LOWPASS_CUTOFF_HZ, btype="lowpass", fs=FS_HZ)
    return sig.lfilter(b, a, np.asarray(x, dtype=float))


def db10(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.asarray(x, dtype=float) + EPS)


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.asarray(x, dtype=float) + EPS)


def nan_stats(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(~np.isfinite(x)):
        return np.nan, np.nan
    return float(np.nanmean(x)), float(np.nanstd(x))


def compute_power_metrics(S: dict, noise_id: str, signal_id: str):
    nf, noise_amp = get_signal(S, noise_id, POWER_SIGNAL_KIND, POWER_SIGNAL_NAME)
    sf, sig_amp = get_signal(S, signal_id, POWER_SIGNAL_KIND, POWER_SIGNAL_NAME)

    nf2, noise_lp = regularize_with_short_gap_fill(nf, noise_amp)
    sf2, sig_lp = regularize_with_short_gap_fill(sf, sig_amp)
    noise_lp = lowpass_lfilter(np.nan_to_num(noise_lp, nan=np.nanmedian(noise_lp)))
    sig_lp = lowpass_lfilter(np.nan_to_num(sig_lp, nan=np.nanmedian(sig_lp)))

    frames, sig_aligned, noise_aligned = align_pair(sf2, sig_lp, nf2, noise_lp)
    sig_power = sig_aligned ** 2
    noise_power = noise_aligned ** 2
    snr_linear = sig_power / (noise_power + EPS)
    snr_db = db10(snr_linear)

    return {
        "frames": frames,
        "signal_lp_amp": sig_aligned,
        "noise_lp_amp": noise_aligned,
        "signal_power": sig_power,
        "noise_power": noise_power,
        "snr_linear": snr_linear,
        "snr_db": snr_db,
    }


def compute_phase_metrics(S: dict, noise_id: str, signal_id: str, kind: str, name: str):
    nf, noise_phase = get_signal(S, noise_id, kind, name)
    sf, sig_phase = get_signal(S, signal_id, kind, name)
    frames, sig_aligned, noise_aligned = align_pair(sf, np.unwrap(sig_phase), nf, np.unwrap(noise_phase))

    phase_diff_rad = sig_aligned - noise_aligned
    abs_phase_ratio = np.abs(sig_aligned) / (np.abs(noise_aligned) + EPS)
    phase_snr_db = db20(abs_phase_ratio)

    return {
        "frames": frames,
        "signal_phase_rad": sig_aligned,
        "noise_phase_rad": noise_aligned,
        "phase_diff_rad": phase_diff_rad,
        "abs_phase_ratio": abs_phase_ratio,
        "phase_snr_db": phase_snr_db,
    }


def add_stats_row(rows: list, label: str, x: np.ndarray):
    mean, std = nan_stats(x)
    rows.append((label, mean, std))


def format_num(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-3):
        return f"{x:.3e}"
    return f"{x:.5g}"


def print_stats(rows: list[tuple[str, float, float]]):
    print("\n=== METRICS mean/std ===")
    width = max([len(r[0]) for r in rows] + [8])
    print(f"{'metric'.ljust(width)} | mean | std")
    print(f"{'-' * width}-|------|------")
    for label, mean, std in rows:
        print(f"{label.ljust(width)} | {format_num(mean)} | {format_num(std)}")


# =============================================================================
# Plotting
# =============================================================================

def thesis_style():
    plt.rcParams.update({
        "figure.figsize": (8.0, 5.5),
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.15,
    })


def savefig(fig, name: str):
    if not SAVE_FIGURES:
        return
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")


def plot_power_metrics(signal_id: str, M: dict):
    f = M["frames"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8.0, 7.0))
    fig.suptitle(f"Power SNR: signal ID {signal_id} vs noise ID {NOISE_TRACK_ID}")

    ax[0].plot(f, M["signal_lp_amp"], label=f"signal {signal_id} LP amp")
    ax[0].plot(f, M["noise_lp_amp"], label=f"noise {NOISE_TRACK_ID} LP amp")
    ax[0].set_ylabel("amplitude")
    ax[0].legend()

    ax[1].plot(f, M["signal_power"], label="signal power")
    ax[1].plot(f, M["noise_power"], label="noise power")
    ax[1].set_ylabel("power")
    ax[1].legend()

    ax[2].plot(f, M["snr_db"], color="C3", label="log SNR")
    ax[2].axhline(0, color="k", alpha=0.3, linewidth=0.8)
    ax[2].set_xlabel("frame")
    ax[2].set_ylabel("SNR [dB]")
    ax[2].legend()

    fig.tight_layout()
    savefig(fig, f"power_snr_signal_{signal_id}_noise_{NOISE_TRACK_ID}")


def plot_phase_metrics(signal_id: str, kind: str, name: str, M: dict):
    f = M["frames"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8.0, 7.0))
    fig.suptitle(f"Phase SNR {kind}:{name}: signal ID {signal_id} vs noise ID {NOISE_TRACK_ID}")

    ax[0].plot(f, M["signal_phase_rad"], label=f"signal {signal_id}")
    ax[0].plot(f, M["noise_phase_rad"], label=f"noise {NOISE_TRACK_ID}")
    ax[0].set_ylabel("phase [rad]")
    ax[0].legend()

    ax[1].plot(f, M["phase_diff_rad"], color="C2", label="signal - noise")
    ax[1].axhline(0, color="k", alpha=0.3, linewidth=0.8)
    ax[1].set_ylabel("difference [rad]")
    ax[1].legend()

    ax[2].plot(f, M["phase_snr_db"], color="C3", label="phase log ratio")
    ax[2].axhline(0, color="k", alpha=0.3, linewidth=0.8)
    ax[2].set_xlabel("frame")
    ax[2].set_ylabel("ratio [dB]")
    ax[2].legend()

    fig.tight_layout()
    safe_name = name.replace("/", "_").replace(" ", "_")
    savefig(fig, f"phase_snr_{kind}_{safe_name}_signal_{signal_id}_noise_{NOISE_TRACK_ID}")


def print_available(S: dict, metadata: dict):
    print("Loaded:", INPUT_NPZ)
    print("Data nickname:", metadata.get("data_nickname"))
    print("Metadata selected_track_ids:", metadata.get("selected_track_ids"))
    print("Available tracks:", list(S.keys()))
    for tid, T in S.items():
        print(f"\nTrack {tid}")
        print("  frames:", len(T["frames"]), f"({T['frames'][0]}..{T['frames'][-1]})" if len(T["frames"]) else "empty")
        print("  raw: ", sorted(T["raw"].keys()))
        print("  line:", sorted(T["line"].keys()))


# =============================================================================
# Main
# =============================================================================

def main():
    thesis_style()
    S, metadata = build_bank(INPUT_NPZ)
    print_available(S, metadata)

    rows = []

    # Baseline stats for chosen raw/line source signals.
    for tid in [NOISE_TRACK_ID] + SIGNAL_TRACK_IDS:
        if tid not in S:
            raise KeyError(f"Track {tid} missing. Available tracks: {list(S)}")

        if POWER_SIGNAL_NAME in S[tid][POWER_SIGNAL_KIND]:
            add_stats_row(rows, f"track {tid} {POWER_SIGNAL_KIND}:{POWER_SIGNAL_NAME}", S[tid][POWER_SIGNAL_KIND][POWER_SIGNAL_NAME])

        for kind, name in PHASE_SIGNALS:
            if name in S[tid][kind]:
                add_stats_row(rows, f"track {tid} {kind}:{name}", S[tid][kind][name])

    # Pairwise SNR metrics.
    for signal_id in SIGNAL_TRACK_IDS:
        P = compute_power_metrics(S, NOISE_TRACK_ID, signal_id)
        add_stats_row(rows, f"signal {signal_id} power_snr_linear", P["snr_linear"])
        add_stats_row(rows, f"signal {signal_id} power_snr_db", P["snr_db"])
        add_stats_row(rows, f"signal {signal_id} lp_signal_amp", P["signal_lp_amp"])
        add_stats_row(rows, f"signal {signal_id} lp_noise_amp", P["noise_lp_amp"])
        plot_power_metrics(signal_id, P)

        for kind, name in PHASE_SIGNALS:
            M = compute_phase_metrics(S, NOISE_TRACK_ID, signal_id, kind, name)
            label = f"signal {signal_id} {kind}:{name}"
            add_stats_row(rows, f"{label} phase_diff_rad", M["phase_diff_rad"])
            add_stats_row(rows, f"{label} abs_phase_ratio", M["abs_phase_ratio"])
            add_stats_row(rows, f"{label} phase_snr_db", M["phase_snr_db"])
            plot_phase_metrics(signal_id, kind, name, M)

    print_stats(rows)
    plt.show()


if __name__ == "__main__":
    main()
