"""
inspect_vitals_export.py
=========================

Standalone viewer for NPZ files exported from visuals/VitalExtraction_plot.py.

Edit INPUT_NPZ below, then run:
    python inspect_vitals_export.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# EDIT THIS
# ---------------------------------------------------------------------------
INPUT_NPZ = "exportedTrackingData/21_michal_auto_radar_horizontal_standing_vitals_export.npz"

INPUT_NPZ = "exportedTrackingData/31_Kofola_big_drive_vitals_export_looongWindows.npz"
INPUT_NPZ = "exportedTrackingData/31_Kofola_big_drive_vitals_export_normalWindows.npz"

frame_start, frame_end = 0,3500
frame_start, frame_end = 0,-1

RAW_SIGNAL_SKIP = {"frames"}
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_RESET = "\033[0m"


def load_export(path: str):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return data, metadata


def collect_groups(data: np.lib.npyio.NpzFile, prefix: str):
    """Return nested dict: group_name -> track_id -> ndarray."""
    groups = defaultdict(dict)
    for key in data.files:
        if not key.startswith(prefix + "/"):
            continue
        parts = key.split("/")
        if prefix == "tracks" and len(parts) >= 3:
            _, track_id, signal_name = parts[:3]
            groups[signal_name][track_id] = np.asarray(data[key])[frame_start:frame_end]
        elif prefix == "lines" and len(parts) >= 3:
            _, track_id, line_name = parts[:3]
            groups[line_name][track_id] = np.asarray(data[key])[frame_start:frame_end]
    return groups


def collect_frames(data: np.lib.npyio.NpzFile):
    frames = {}
    for key in data.files:
        if key.startswith("tracks/") and key.endswith("/frames"):
            _, track_id, _ = key.split("/", 2)
            frames[track_id] = np.asarray(data[key])[frame_start:frame_end] #frame_start, frame_end
    return frames


def pretty_line_settings(metadata: dict, line_name_safe: str) -> str:
    if line_name_safe == "Line_0":
        return "Line 0: unwrapped phase"
    line_name = line_name_safe.replace("_", " ")
    active = metadata.get("active_lines", {})
    settings = active.get(line_name, {})
    if not settings:
        return line_name
    method = settings.get("Method", "?")
    input_name = settings.get("Input", "?")
    extras = []
    for key, val in settings.items():
        if key in {"Plot 1", "Plot 2", "Method", "Input"}:
            continue
        extras.append(f"{key}={val}")
    extra_text = ", ".join(extras)
    if extra_text:
        return f"{line_name}: {method}, input={input_name}, {extra_text}"
    return f"{line_name}: {method}, input={input_name}"


def plot_grouped_signals(groups, frames_by_track, title: str, metadata: dict | None = None):
    names = [name for name in groups.keys() if name not in RAW_SIGNAL_SKIP]
    if not names:
        print(f"No signals to plot for {title}")
        return

    fig, axes = plt.subplots(len(names), 1, sharex=False, figsize=(13, max(3, 2.7 * len(names))))
    if len(names) == 1:
        axes = [axes]
    fig.suptitle(title)

    for ax, name in zip(axes, names):
        tracks = groups[name]
        for track_id, values in tracks.items():
            x = frames_by_track.get(track_id, np.arange(len(values)))
            n = min(len(x), len(values))
            if n == 0:
                continue
            ax.plot(x[:n], values[:n], label=f"ID {track_id}")
        ax.grid(True, alpha=0.3)
        if title.lower().startswith("line") and metadata is not None:
            ax.set_title(pretty_line_settings(metadata, name), fontsize=8)
        else:
            ax.set_title(name, fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("Frame")
    fig.tight_layout()


def format_num(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-3):
        return f"{x:.3e}"
    return f"{x:.4g}"


def make_stats_rows(groups, frames_by_track, prefix: str):
    rows = []
    for signal_name, tracks in groups.items():
        if signal_name in RAW_SIGNAL_SKIP:
            continue
        for track_id, values in tracks.items():
            arr = np.asarray(values, dtype=float)[frame_start:frame_end]
            if arr.size == 0:
                mean, std = np.nan, np.nan
            else:
                mean, std = float(np.nanmean(arr)), float(np.nanstd(arr))
            rows.append((f"{prefix}:{signal_name}", track_id, mean, std))
    return rows


def print_markdown_stats(rows):
    if not rows:
        print("No stats.")
        return

    signals = []
    tracks = []
    by_key = {}
    for signal, track, mean, std in rows:
        if signal not in signals:
            signals.append(signal)
        if track not in tracks:
            tracks.append(track)
        by_key[(signal, track)] = (mean, std)

    headers = ["Signal"] + [f"ID {t}" for t in tracks]
    table = []
    for signal in signals:
        row = [signal]
        for track in tracks:
            mean, std = by_key.get((signal, track), (np.nan, np.nan))
            row.append(f"μ={format_num(mean)}, σ={format_num(std)}")
        table.append(row)

    widths = [len(h) for h in headers]
    for row in table:
        widths = [max(w, len(str(cell))) for w, cell in zip(widths, row)]

    def fmt_row(row):
        return "| " + " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)) + " |"

    print(f"\n{ANSI_BOLD}{ANSI_CYAN}Signal statistics: mean/std{ANSI_RESET}")
    print(fmt_row(headers))
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in table:
        print(fmt_row(row))


def plot_stats_table(rows):
    if not rows:
        return
    signals = []
    tracks = []
    by_key = {}
    for signal, track, mean, std in rows:
        if signal not in signals:
            signals.append(signal)
        if track not in tracks:
            tracks.append(track)
        by_key[(signal, track)] = (mean, std)

    cell_text = []
    for signal in signals:
        row = []
        for track in tracks:
            mean, std = by_key.get((signal, track), (np.nan, np.nan))
            row.append(f"μ={format_num(mean)}\nσ={format_num(std)}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(tracks)), max(4, 0.45 * len(signals))))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=signals,
        colLabels=[f"ID {t}" for t in tracks],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("Signal statistics: mean/std")
    fig.tight_layout()


def main():
    path = Path(INPUT_NPZ)
    if not path.exists():
        raise FileNotFoundError(f"Edit INPUT_NPZ at top of script. Missing: {path}")

    data, metadata = load_export(str(path))
    print(f"Loaded: {path}")
    print(f"Format: {metadata.get('format')}")
    print(f"Data nickname: {metadata.get('data_nickname')}")
    print(f"Tracks: {metadata.get('selected_track_ids')}")

    frames_by_track = collect_frames(data)
    raw_groups = collect_groups(data, "tracks")
    line_groups = collect_groups(data, "lines")

    plot_grouped_signals(raw_groups, frames_by_track, "Raw tracked signals")
    plot_grouped_signals(line_groups, frames_by_track, "Line outputs", metadata=metadata)

    rows = []
    rows.extend(make_stats_rows(raw_groups, frames_by_track, "raw"))
    rows.extend(make_stats_rows(line_groups, frames_by_track, "line"))
    print_markdown_stats(rows)
    plot_stats_table(rows)

    plt.show()


if __name__ == "__main__":
    main()
