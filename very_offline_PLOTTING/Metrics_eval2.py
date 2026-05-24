import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np


def load_export_signals(path, frame_limit=None, track_ids=None):
    """
    Load exported tracking NPZ into simple S dict.

    Returns:
        S[track_id]["frames"]
        S[track_id]["raw"][signal_name]
        S[track_id]["line"][line_name]
        metadata

    Example:
        S, meta = load_export_signals("exportedTrackingData/file.npz", frame_limit=(0, 3500))

        frames = S["1"]["frames"]
        phase = S["1"]["line"]["Line_0"]
        power = S["1"]["raw"]["power_high_dop"]
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

    # First collect frame supports.
    for key in data.files:
        if key.startswith("tracks/") and key.endswith("/frames"):
            _, track_id, _ = key.split("/", 2)

            if track_ids is not None and track_id not in track_ids:
                continue

            frames = np.asarray(data[key], dtype=int)
            mask = keep_frame_mask(frames)

            S[track_id]["frames"] = frames[mask]

    # Then collect raw track signals.
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

    # Then collect line signals.
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


def repair_supports(S, signals, max_gap=8, mode="intersection"):
    """
    Align selected signals onto a common frame support.

    Missing holes up to `max_gap` frames are linearly interpolated.
    Bigger holes stay invalid and are removed from comparison.

    Args:
        S:
            dict from load_export_signals()

        signals:
            list of tuples:
                (track_id, kind, signal_name)

            where kind is:
                "raw" or "line"

            Example:
                [
                    ("0", "raw", "power_high_dop"),   # noise
                    ("1", "raw", "power_high_dop"),   # signal
                    ("2", "raw", "power_high_dop"),
                ]

        max_gap:
            maximum missing-frame hole to repair by interpolation

        mode:
            "intersection" = only frames valid for all signals
            "union" = all frames, missing values stay np.nan

    Returns:
        frames_common, repaired

        repaired[(track_id, kind, signal_name)] = aligned_values

    Example:
        frames, R = repair_supports(S, [
            ("0", "line", "Line_0"),
            ("1", "line", "Line_0"),
        ])

        noise_phase = R[("0", "line", "Line_0")]
        signal_phase = R[("1", "line", "Line_0")]
        phase_diff = signal_phase - noise_phase
    """

    def regularize(frames, values):
        frames = np.asarray(frames, dtype=int)
        values = np.asarray(values, dtype=float)

        order = np.argsort(frames)
        frames = frames[order]
        values = values[order]

        # Remove duplicate frame indices, keeping last occurrence.
        _, unique_last_indices = np.unique(frames[::-1], return_index=True)
        keep = len(frames) - 1 - unique_last_indices
        keep = np.sort(keep)

        frames = frames[keep]
        values = values[keep]

        full_frames = np.arange(frames[0], frames[-1] + 1)
        full_values = np.full(len(full_frames), np.nan)

        frame_to_index = frames - frames[0]
        full_values[frame_to_index] = values

        # Fill only short holes.
        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]
            gap = f1 - f0 - 1

            if 0 < gap <= max_gap:
                idx0 = f0 - frames[0]
                idx1 = f1 - frames[0]

                full_values[idx0:idx1 + 1] = np.linspace(
                    values[i],
                    values[i + 1],
                    gap + 2,
                )

        return full_frames, full_values

    regularized = {}

    for track_id, kind, name in signals:
        track_id = str(track_id)

        frames = S[track_id]["frames"]
        values = S[track_id][kind][name]

        full_frames, full_values = regularize(frames, values)
        regularized[(track_id, kind, name)] = (full_frames, full_values)

    start = min(fr[0] for fr, _ in regularized.values())
    end = max(fr[-1] for fr, _ in regularized.values())

    if mode == "intersection":
        start = max(fr[0] for fr, _ in regularized.values())
        end = min(fr[-1] for fr, _ in regularized.values())

    frames_common = np.arange(start, end + 1)

    repaired = {}

    for key, (frames, values) in regularized.items():
        aligned = np.full(len(frames_common), np.nan)

        overlap_start = max(frames_common[0], frames[0])
        overlap_end = min(frames_common[-1], frames[-1])

        if overlap_end >= overlap_start:
            dst0 = overlap_start - frames_common[0]
            dst1 = overlap_end - frames_common[0] + 1

            src0 = overlap_start - frames[0]
            src1 = overlap_end - frames[0] + 1

            aligned[dst0:dst1] = values[src0:src1]

        repaired[key] = aligned

    if mode == "intersection":
        valid = np.ones(len(frames_common), dtype=bool)
        for values in repaired.values():
            valid &= np.isfinite(values)

        frames_common = frames_common[valid]
        repaired = {key: values[valid] for key, values in repaired.items()}

    return frames_common, repaired

FS_HZ = 20

def track_time_or_frames(S: dict, track_id: str, use_seconds: bool = False) -> np.ndarray:
    frames = np.asarray(S[str(track_id)]["frames"])
    if use_seconds:
        return (frames - frames[0]) / FS_HZ
    return frames

def simpleSNR_20Log(noise,signal):
    snr_db = 20 * np.log10(np.abs(signal) / (np.abs(noise) + 1e-12))
    return snr_db

def simpleSNR_normal(noise,signal):
    snr = 20 * np.abs(signal) / (np.abs(noise) + 1e-12) #np.log10(np.abs(signal) / (np.abs(noise) + 1e-12))
    return snr

def phase_rad2mm(sig_rad):
    lambda_mm = 5
    sig_mm = lambda_mm  * sig_rad/(4*np.pi)
    return sig_mm

S, meta = load_export_signals(
    "exportedTrackingData/21_michal_auto_radar_horizontal_standing_vitals_export.npz",
    frame_limit=(0, 3500),
)

signals = ["A","B"]
noise = "RE"

frames, R = repair_supports(S, [
    ("RE", "raw", "power_high_dop"),   # noise
    ("A", "raw", "power_high_dop"),   # signal
    ("B", "raw", "power_high_dop"),   # signal

    ("RE", "line", "Line_2"),   # noise
    ("A", "line", "Line_2"),   # signal
    ("B", "line", "Line_2"),   # signal

    ("RE", "line", "Line_5"),   # noise
    ("A", "line", "Line_5"),   # signal
    ("B", "line", "Line_5"),   # signal
])



t_A = track_time_or_frames(R,"A",use_seconds=True)
t_B = track_time_or_frames(R,"B",use_seconds=True)
t_RE = track_time_or_frames(R,"RE",use_seconds=True)


sig_A_pwr = S["A"]["raw"]["power_all"]
sig_RE_pwr = S["RE"]["raw"]["power_all"]
sig_A_highD = S["A"]["raw"]["power_high_dop"]
sig_RE_highD = S["RE"]["raw"]["power_high_dop"]
    
sig_A_detrend =  S["A"]["line"]["Line_2"]
sig_RE_detrend =  S["RE"]["line"]["Line_2"]

sig_A_detrend = phase_rad2mm(sig_A_detrend)
sig_RE_detrend = phase_rad2mm(sig_RE_detrend)

snr_detrend_A_log = simpleSNR_20Log(noise=sig_RE_detrend,signal=sig_A_detrend)
snr_detrend_A_ratio = simpleSNR_normal(noise=sig_RE_detrend,signal=sig_A_detrend)


fig, ax = plt.subplots(2, 1 ) #, figsize=(7.2, 3.2))
ax[0].plot(t_A, sig_A_detrend, label="A")
ax[0].plot(t_B, sig_A_detrend, label="B")
ax[0].plot(t_RE, sig_RE_detrend, label="RE")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("displacement [mm]")
ax[0].set_title("Ph. unwrap detrended")
ax[0].legend()

ax[1].plot(t_A, snr_detrend_A_ratio, label="A")
# ax[0].plot(t_B, sig_A_detrend, label="B")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("displacement [mm]")
ax[1].set_title("SNR linear")
ax[1].legend()


ax[2].plot(t_A, snr_detrend_A_log, label="A")
# ax[0].plot(t_B, sig_A_detrend, label="B")
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("log [dB]")
ax[2].set_title("SNR log")
ax[2].legend()

for a in ax:
    a.legend().set_draggable(True)

fig.tight_layout(h_pad=0.8)

plt.show()



# ax[1].plot(t_A, sig_A_pwr, label="A")
# ax[1].plot(t_RE, sig_RE_pwr, label="RE")
# # ax[1].plot(t_A, sig_A_inst_A, label="inst. freq")
# ax[1].set_xlabel("time [s]")
# ax[1].set_ylabel("amplitude [1]")
# ax[1].set_title("Reflected signal amplitude, all doppler  bins")
# ax[1].legend()
# ax[1].set_yscale('log')

# ax[2].plot(t_A, sig_A_highD, label="A")
# ax[2].plot(t_RE, sig_RE_highD, label="RE")
# ax[2].set_xlabel("time [s]")
# ax[2].set_ylabel("amplitude [1]")
# ax[2].set_title("Reflected signal amplitude, higher doppler bins")
# ax[2].legend()
# ax[2].set_yscale('log')

# for a in ax:
#     a.legend().set_draggable(True)