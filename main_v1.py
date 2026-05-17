"""
main_v1.py
==========

Launcher/control panel for HR_explorer.

Edit these sections first:
    - FILE_PRESETS: known NPZ files, nicknames, per-file parameter overrides
    - AUTORUN: what loads/processes/opens when the app starts
    - BASE_PARAMS / NORMAL_PROCESS_OVERRIDES / MESH_PROCESS_OVERRIDES

Run from the HR_explorer directory:
    python main_v1.py
"""

from __future__ import annotations

import ast
import copy
import os
import sys
from pathlib import Path
from typing import Callable, Any

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QMessageBox,
    QScrollArea,
    QFrame,
)

from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process
import processing.resampling as resampling

import visuals.DoA_3Dmesh as DoA_3Dmesh
import visuals.PhaseUnWr_plot_2 as PhaseUnWr_plot
import visuals.HR_STFT_plot_2 as HR_STFT_plot
import visuals.DoA_2DAziPlane as DoA_2DAziPlane
import visuals.DoA_2DAziPlane_tracking as DoA_2DTracking
import visuals.TrackedSignals_plot as TrackingPlot
import visuals.VitalExtraction_plot as VitalsPlot
import visuals.param_controls as pctrl


# ---------------------------------------------------------------------------
# Easy-to-edit hardcoded config
# ---------------------------------------------------------------------------

BASE_PARAMS: dict[str, Any] = {
    "i_Frames_begin": 100,
    "i_Frames_end": 600,
    "i_Range_begin": 5,
    "i_Range_end": 30,
    "i_Doppler_begin": 0,
    "i_Doppler_end": 32,
    "DoA_azi_N_elements": 8,
    "DoA_ele_N_elements": 1,
    "DoA_azi_range_degs": 90.0,
    "DoA_ele_range_degs": 30.0,
    "Channel_processing": "DoA_customFFT",
    "Doppler_processing": "FFT",
    "range_index2dist": 0.046,
    "frame_index2time": 5e-2,
    "doppler_index2freq": 1 / (5.76 * 1e-3),
    "doppler_index2vel": 0.157,
    "radarRotation_deg": 0.0,
}

# Applied after loading a file and after HR_process.defaultSliders(...).
NORMAL_PROCESS_OVERRIDES: dict[str, Any] = {
    "DoA_ele_N_elements": 1,
    "DoA_azi_N_elements": 8,
    "Doppler_processing": "FFT",
}

# DoA_3Dmesh wants a different, heavier penteract. Keep frame count sane.
MESH_PROCESS_OVERRIDES: dict[str, Any] = {
    "DoA_ele_N_elements": 12,
    "DoA_azi_N_elements": 24,
    "Doppler_processing": "None",
    "Channel_processing": "DoA_customFFT",
}

# Known data files. Add/remove entries here. The combobox shows nickname.
FILE_PRESETS: list[dict[str, Any]] = [
    {
        "nickname": "21 michal auto radar horizontal standing",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_21_michalauto_radarvodorovne_stani_24-04-2026_14-32-50.npz",
        "params_override": {
            "enforcement_cages": [
                {
                    "id": "A",
                    "r_begin": 18,
                    "r_end": 22,
                    "azi_begin": 3,
                    "azi_end": 4,
                },
                {
                    "id": "B",
                    "r_begin": 19,
                    "r_end": 23,
                    "azi_begin": 5,
                    "azi_end": 6,
                },
                {
                    "id": "R",
                    "r_begin": 24,
                    "r_end": 26,
                    "azi_begin": 2,
                    "azi_end": 4,
                },
            ]
        },
    },
    {
        "nickname": "22 michal auto radar horizontal driving",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_22_michalauto_radarvodorovne_jizda_rucedolekdyztoslo_24-04-2026_14-39-26.npz",
        "params_override": {
            "enforcement_cages": [
                {
                    "id": "A",
                    "r_begin": 18,
                    "r_end": 22,
                    "azi_begin": 3,
                    "azi_end": 4,
                },
                {
                    "id": "B",
                    "r_begin": 19,
                    "r_end": 23,
                    "azi_begin": 5,
                    "azi_end": 6,
                },
                {
                    "id": "R",
                    "r_begin": 24,
                    "r_end": 26,
                    "azi_begin": 2,
                    "azi_end": 4,
                },
            ]
        },
    },
    {
        "nickname": "16 michal auto radar vertical standing",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_16_michalautoklid_24-04-2026_13-41-17.npz",
        "params_override": {},
    },
    {
        "nickname": "17 michal auto radar vertical standing, engine on",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_17_michalauto_motorbezi_24-04-2026_13-47-15.npz",
        "params_override": {},
    },
    {
        "nickname": "18 michal auto radar vertical drive, hands down",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_18_michalauto_jizda_24-04-2026_14-00-40.npz",
        "params_override": {},
    },
    {
        "nickname": "19 michal auto radar vertical drive, hands up",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_19_michalauto_jizda_rucehore_24-04-2026_14-08-41.npz",
        "params_override": {},
    },
    {
        "nickname": "20 michal auto radar vertical drive, hands up, natural driver movements",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/20 unR_meas_20_michalauto_jizda_rucehore_normalnipohybynavic_24-04-2026_14-15-53.npz ",
        "params_override": {},
    },
    
    {
        "nickname": "25 martinesn 40Hz standing",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_25_martinesn_40hz_zkouskabezbr_stani_07-05-2026_16-52-27.npz",
        "params_override": {"frame_index2time": 1 / 40},
    },
    {
        "nickname": "26 martinesn maybe 20Hz standing",
        "path": "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_26_martinesn_40hz_again_standing_07-05-2026_17-27-52.npz",
        "params_override": {"frame_index2time": 1 / 20},
    },
]

# Startup behavior. This is deliberately dumb and obvious — change it here.
AUTORUN: dict[str, Any] = {
    "enabled": False,
    "file_nickname": "21 michal auto radar horizontal standing",
    "process_normal": True,
    "process_mesh": False,  # expensive; leave False unless you mean it
    "mesh_frame_skip_step": 40,
    "launch_normal_windows": ["Phase Unwrap", "Tracking", "Tracked Signals"],
    "launch_mesh_window": False,
}

# Values shown as dropdowns in DictTableControl instead of free text.
PARAM_OPTIONS: dict[str, list[str]] = {
    "Channel_processing": ["DoA_customFFT", "None"],
    "Doppler_processing": ["FFT", "None"],
}

# DictTableControl treats any list as a combobox. That is great for dropdowns
# and absolutely lethal for list-of-dicts params like enforcement cages.
# Keep these hidden from the generic params editor and preserve them in code.
COMPLEX_PARAM_KEYS = {"enforcement_cages", "Enforced tracks"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_for_editor(params: dict[str, Any]) -> dict[str, Any]:
    """Convert selected string params to pctrl dropdown specs.

    Important: list-of-dicts params are intentionally hidden, because
    DictTableControl interprets every list as combobox options and would turn
    enforcement_cages into a single string. Yes, that was the bug. Sneaky bastard.
    """
    out = {key: value for key, value in params.items() if key not in COMPLEX_PARAM_KEYS}
    for key, options in PARAM_OPTIONS.items():
        if key in out:
            current = str(out[key])
            ordered = [current] + [opt for opt in options if opt != current]
            out[key] = ordered
    return out


def _coerce_complex_param(value: Any) -> Any:
    """Recover complex params if an older editor/session already stringified them."""
    if not isinstance(value, str):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    return parsed


def _restore_complex_params(edited: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """Merge editor scalars back into previous params without losing hidden complex params."""
    merged = dict(previous)
    merged.update(edited)
    for key in COMPLEX_PARAM_KEYS:
        if key in previous:
            merged[key] = _coerce_complex_param(previous[key])
        elif key in merged:
            merged[key] = _coerce_complex_param(merged[key])
    return merged


def _params_from_editor(editor: pctrl.DictTableControl, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    edited = editor.value()
    if previous is None:
        return edited
    return _restore_complex_params(edited, previous)


def _add_doa_points(params: dict[str, Any], doa_dict: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    params["azi_points"] = doa_dict["azi_mesh_range"]
    params["ele_points"] = doa_dict["ele_mesh_range"]
    return params


def _extract_hrs(loaded_data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    hrs = {key: val for key, val in loaded_data.items() if key.startswith("hr_")}
    if hrs:
        hrs = resampling.resample_HR_to_frames(hrs, 1 / params["frame_index2time"])
    return hrs


def _safe_call_update(window: QWidget, method_names: tuple[str, ...], *args: Any) -> bool:
    for name in method_names:
        method = getattr(window, name, None)
        if callable(method):
            method(*args)
            return True
    return False


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HR Explorer launcher v1")
        self.resize(1180, 760)

        self.loaded_data: dict[str, Any] | None = None
        self.frames = None
        self.hrs: dict[str, Any] = {}
        self.data_nickname = "none"
        self.file_bound_overrides: dict[str, Any] = {}

        self.normal_params = copy.deepcopy(BASE_PARAMS)
        self.mesh_params = copy.deepcopy(BASE_PARAMS)
        self.normal_penteract = None
        self.mesh_penteract = None

        self.open_windows: list[dict[str, Any]] = []
        self.window_counts: dict[str, int] = {}
        self.latest_tracking_window: QWidget | None = None
        self.tracking_source_label: QLabel | None = None

        self._build_ui()
        self._load_initial_file_if_possible()
        self._autorun_if_enabled()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.layout_main = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self._build_file_selector(self.layout_main)
        self._build_normal_panel(self.layout_main)
        self._build_mesh_panel(self.layout_main)

        quit_btn = QPushButton("Exit Application")
        quit_btn.clicked.connect(QApplication.instance().quit)  # type: ignore[union-attr]
        outer.addWidget(quit_btn)

    def _build_file_selector(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Data file")
        box.setMaximumHeight(118)
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setVerticalSpacing(4)
        layout.setHorizontalSpacing(8)

        self.file_combo = QComboBox()
        for preset in FILE_PRESETS:
            self.file_combo.addItem(preset["nickname"])
        self.file_combo.currentIndexChanged.connect(self._load_selected_preset)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Custom .npz path")

        browse_btn = QPushButton("Browse")
        browse_btn.setMaximumWidth(90)
        browse_btn.clicked.connect(self._browse_custom_file)

        load_custom_btn = QPushButton("Load custom")
        load_custom_btn.setMaximumWidth(110)
        load_custom_btn.clicked.connect(self._load_custom_path)

        self.file_status = QLabel("No file loaded")
        self.file_status.setWordWrap(False)

        layout.addWidget(QLabel("Known:"), 0, 0)
        layout.addWidget(self.file_combo, 0, 1, 1, 3)
        layout.addWidget(QLabel("Other:"), 1, 0)
        layout.addWidget(self.path_edit, 1, 1)
        layout.addWidget(browse_btn, 1, 2)
        layout.addWidget(load_custom_btn, 1, 3)
        layout.addWidget(self.file_status, 2, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        parent.addWidget(box)

    def _build_normal_panel(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Normal processing / plot windows")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.normal_param_editor = pctrl.DictTableControl(
            "Normal process_A params",
            _params_for_editor(self.normal_params),
            orientation="horizontal",
        )
        layout.addWidget(self.normal_param_editor)

        lower = QHBoxLayout()
        lower.setSpacing(12)

        process_col = QVBoxLayout()
        process_col.setSpacing(4)
        process_btn = QPushButton("Process normal data")
        process_btn.clicked.connect(self.process_normal_data)
        update_btn = QPushButton("Update opened windows")
        update_btn.clicked.connect(self.update_open_normal_windows)
        process_col.addWidget(process_btn)
        process_col.addWidget(update_btn)

        self.tracking_source_label = QLabel("Tracking source: none")
        self.tracking_source_label.setWordWrap(False)
        process_col.addWidget(self.tracking_source_label)

        tracked_btn = QPushButton("Launch Tracked Signals")
        tracked_btn.clicked.connect(lambda: self.launch_normal_window("Tracked Signals"))
        vitals_btn = QPushButton("Launch Vitals from current tracking")
        vitals_btn.clicked.connect(self.launch_vitals_from_current_tracking)
        process_col.addWidget(tracked_btn)
        process_col.addWidget(vitals_btn)
        process_col.addStretch(1)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)

        launch_grid = QGridLayout()
        launch_grid.setHorizontalSpacing(4)
        launch_grid.setVerticalSpacing(4)
        for i, name in enumerate(self._normal_window_names()):
            btn = QPushButton(name)
            btn.setMaximumHeight(26)
            btn.clicked.connect(lambda _checked=False, n=name: self.launch_normal_window(n))
            launch_grid.addWidget(btn, i // 2, i % 2)

        lower.addLayout(process_col, 1)
        lower.addWidget(divider)
        lower.addLayout(launch_grid, 2)
        layout.addLayout(lower)

        self.normal_status = QLabel("Normal penteract: not processed")
        self.normal_status.setWordWrap(False)
        layout.addWidget(self.normal_status)
        parent.addWidget(box)

    def _build_mesh_panel(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("DoA_3Dmesh processing / window")
        layout = QVBoxLayout(box)

        self.mesh_param_editor = pctrl.DictTableControl(
            "DoA_3Dmesh process_A params",
            _params_for_editor(self.mesh_params),
            orientation="horizontal",
        )
        layout.addWidget(self.mesh_param_editor)

        skip_row = QHBoxLayout()
        skip_row.addWidget(QLabel("Frame skip step:"))
        self.mesh_skip_spin = QSpinBox()
        self.mesh_skip_spin.setRange(1, 10_000)
        self.mesh_skip_spin.setValue(int(AUTORUN.get("mesh_frame_skip_step", 10)))
        skip_row.addWidget(self.mesh_skip_spin)
        layout.addLayout(skip_row)

        btn_row = QHBoxLayout()
        process_btn = QPushButton("Process DoA_3Dmesh data")
        process_btn.clicked.connect(self.process_mesh_data)
        launch_btn = QPushButton("Launch DoA_3Dmesh")
        launch_btn.clicked.connect(self.launch_mesh_window)
        update_btn = QPushButton("Update opened DoA_3Dmesh windows")
        update_btn.clicked.connect(self.update_open_mesh_windows)
        btn_row.addWidget(process_btn)
        btn_row.addWidget(launch_btn)
        btn_row.addWidget(update_btn)
        layout.addLayout(btn_row)

        self.mesh_status = QLabel("Mesh penteract: not processed")
        layout.addWidget(self.mesh_status)
        parent.addWidget(box)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def _load_initial_file_if_possible(self) -> None:
        nickname = AUTORUN.get("file_nickname")
        idx = next((i for i, p in enumerate(FILE_PRESETS) if p["nickname"] == nickname), 0)
        if FILE_PRESETS:
            self.file_combo.setCurrentIndex(idx)
            self._load_selected_preset(idx)

    def _load_selected_preset(self, idx: int) -> None:
        if idx < 0 or idx >= len(FILE_PRESETS):
            return
        preset = FILE_PRESETS[idx]
        self._load_file(
            path=preset["path"],
            nickname=preset["nickname"],
            params_override=preset.get("params_override", {}),
        )

    def _browse_custom_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load NPZ data", "", "NPZ files (*.npz);;All files (*)")
        if path:
            self.path_edit.setText(path)
            self._load_custom_path()

    def _load_custom_path(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            return
        nickname = Path(path).stem
        self._load_file(path=path, nickname=nickname, params_override={})

    def _load_file(self, path: str, nickname: str, params_override: dict[str, Any]) -> None:
        try:
            loaded = loadNPZ(path)
            frames = loaded["frames"]
        except Exception as exc:  # noqa: BLE001 - GUI should show the actual failure
            QMessageBox.critical(self, "Failed to load NPZ", f"{path}\n\n{exc}")
            self.file_status.setText(f"Load failed: {path}")
            return

        self.loaded_data = loaded
        self.frames = frames
        self.hrs = _extract_hrs(loaded, {**BASE_PARAMS, **params_override})
        self.data_nickname = nickname
        self.file_bound_overrides = dict(params_override)

        self.normal_params = self._fresh_params(NORMAL_PROCESS_OVERRIDES)
        self.mesh_params = self._fresh_params(MESH_PROCESS_OVERRIDES)
        self.normal_param_editor.set_newParams(_params_for_editor(self.normal_params))
        self.mesh_param_editor.set_newParams(_params_for_editor(self.mesh_params))

        self.normal_penteract = None
        self.mesh_penteract = None
        self.file_status.setText(f"Loaded: {nickname}  |  {path}  |  frames shape: {frames.shape}")
        self.normal_status.setText("Normal penteract: not processed for current file")
        self.mesh_status.setText("Mesh penteract: not processed for current file")

    def _fresh_params(self, process_overrides: dict[str, Any]) -> dict[str, Any]:
        if self.frames is None:
            return copy.deepcopy(BASE_PARAMS)
        params = copy.deepcopy(BASE_PARAMS)
        params = HR_process.defaultSliders(self.frames, params)
        params.update(process_overrides)
        params.update(self.file_bound_overrides)
        return params

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_normal_data(self) -> None:
        if self.frames is None:
            QMessageBox.warning(self, "No data", "Load an NPZ file first.")
            return
        self.normal_params = _params_from_editor(self.normal_param_editor, self.normal_params)
        self.normal_status.setText("Normal penteract: processing...")
        QApplication.processEvents()

        penteract, doa_dict = HR_process.process_A(self.frames, self.normal_params)
        self.normal_params = _add_doa_points(self.normal_params, doa_dict)
        self.normal_penteract = penteract
        self.normal_status.setText(f"Normal penteract ready: {penteract.shape} for {self.data_nickname}")

    def process_mesh_data(self) -> None:
        if self.frames is None:
            QMessageBox.warning(self, "No data", "Load an NPZ file first.")
            return
        self.mesh_params = _params_from_editor(self.mesh_param_editor, self.mesh_params)
        step = max(1, int(self.mesh_skip_spin.value()))

        # First apply the user's selected frame window, then decimate it.
        # The resulting penteract has local frame indices 0..N after skipping.
        frame_begin = int(self.mesh_params.get("i_Frames_begin", 0))
        frame_end = int(self.mesh_params.get("i_Frames_end", self.frames.shape[0]))
        mesh_frames = self.frames[frame_begin:frame_end:step]

        params = dict(self.mesh_params)
        params = HR_process.defaultSliders(mesh_frames, params)

        # Preserve user-selected range/doppler and non-dimension params.
        # Frame begin/end intentionally become 0..len(mesh_frames), because the data is sliced already.
        for key, value in self.mesh_params.items():
            if key not in {"i_Frames_begin", "i_Frames_end"}:
                params[key] = value
        params["i_Frames_begin"] = 0
        params["i_Frames_end"] = mesh_frames.shape[0]
        params["frame_index2time"] = self.mesh_params["frame_index2time"] * step
        params["_frame_skip_step"] = step
        params["_source_i_Frames_begin"] = frame_begin
        params["_source_i_Frames_end"] = frame_end

        self.mesh_status.setText(f"Mesh penteract: processing every {step}. frame...")
        QApplication.processEvents()

        penteract, doa_dict = HR_process.process_A(mesh_frames, params)
        self.mesh_params = _add_doa_points(params, doa_dict)
        self.mesh_penteract = penteract
        self.mesh_status.setText(f"Mesh penteract ready: {penteract.shape} for {self.data_nickname} (step {step})")

    # ------------------------------------------------------------------
    # Window launching / updating
    # ------------------------------------------------------------------
    def _normal_window_names(self) -> list[str]:
        return ["Phase Unwrap", "HR STFT", "DoA 2D Azi Plane", "Tracking"]

    def _normal_specs(self) -> dict[str, Callable[[], QWidget]]:
        return {
            "Phase Unwrap": self._new_phase_window,
            "HR STFT": self._new_stft_window,
            "DoA 2D Azi Plane": self._new_doa2d_window,
            "Tracking": self._new_tracking_window,
            "Tracked Signals": self._new_tracked_signals_window,
            "Vitals": self._new_vitals_window,
        }

    def launch_normal_window(self, name: str) -> None:
        if self.normal_penteract is None:
            self.process_normal_data()
            if self.normal_penteract is None:
                return
        window = self._normal_specs()[name]()
        self._register_window(window, name, panel="normal")
        if name == "Tracking":
            self.latest_tracking_window = window
            self._refresh_tracking_source_label()
        window.show()

    def launch_mesh_window(self) -> None:
        if self.mesh_penteract is None:
            self.process_mesh_data()
            if self.mesh_penteract is None:
                return
        data4d = self.mesh_penteract[:, 0, :, :, :]
        window = DoA_3Dmesh.MeshPlotter3D(data4d, self._window_params("DoA_3Dmesh", self.mesh_params))
        self._register_window(window, "DoA_3Dmesh", panel="mesh")
        window.show()

    def update_open_normal_windows(self) -> None:
        if self.normal_penteract is None:
            self.process_normal_data()
        for entry in list(self.open_windows):
            if entry["panel"] == "normal" and not entry["window"].isHidden():
                self._update_normal_window(entry["window"], entry["name"])

    def update_open_mesh_windows(self) -> None:
        if self.mesh_penteract is None:
            self.process_mesh_data()
        if self.mesh_penteract is None:
            return
        for entry in list(self.open_windows):
            if entry["panel"] == "mesh" and not entry["window"].isHidden():
                data4d = self.mesh_penteract[:, 0, :, :, :]
                _safe_call_update(entry["window"], ("update_newData", "update_NewData", "update_onNewData"), data4d, self._window_params(entry["name"], self.mesh_params))

    def _register_window(self, window: QWidget, name: str, panel: str) -> None:
        self.window_counts[name] = self.window_counts.get(name, 0) + 1
        number = self.window_counts[name]
        title = f"{name} {number} - data {self.data_nickname}"
        window.setWindowTitle(title)
        self.open_windows.append({"window": window, "name": name, "panel": panel, "number": number})

    def _tracking_source_text(self) -> str:
        if self.latest_tracking_window is None or self.latest_tracking_window.isHidden():
            return "Tracking source: none"
        return f"Tracking source: {self.latest_tracking_window.windowTitle()}"

    def _refresh_tracking_source_label(self) -> None:
        if self.tracking_source_label is not None:
            self.tracking_source_label.setText(self._tracking_source_text())

    def launch_vitals_from_current_tracking(self) -> None:
        if self.latest_tracking_window is None or self.latest_tracking_window.isHidden():
            QMessageBox.warning(self, "No tracking source", "Launch/select a Tracking window first.")
            self._refresh_tracking_source_label()
            return
        window = VitalsPlot.PlotWindow()
        self._update_normal_window(window, "Vitals")
        self._register_window(window, "Vitals", panel="normal")
        window.show()

    def _window_params(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        out = dict(params)
        number = self.window_counts.get(name, 0) + 1
        out["_window_title"] = f"{name} {number} - data {self.data_nickname}"
        out["_data_nickname"] = self.data_nickname
        out["_window_name"] = name
        out["_window_number"] = number
        return out

    def _update_normal_window(self, window: QWidget, name: str) -> None:
        if self.normal_penteract is None:
            return
        params = self._window_params(name, self.normal_params)
        if name == "Phase Unwrap":
            _safe_call_update(window, ("update_newData", "update_NewData", "update_onNewData"), self.normal_penteract, params)
        elif name == "HR STFT":
            _safe_call_update(window, ("update_newData", "update_NewData", "update_onNewData"), self.normal_penteract[:, 0, :, :, :], params)
        elif name == "DoA 2D Azi Plane":
            _safe_call_update(window, ("update_newData", "update_NewData", "update_onNewData"), self.normal_penteract, params)
        elif name == "Tracking":
            _safe_call_update(window, ("update_newData", "update_NewData", "update_onNewData"), self.normal_penteract, params)
            self.latest_tracking_window = window
            self._refresh_tracking_source_label()
        elif name == "Tracked Signals":
            if self.latest_tracking_window is not None:
                window.assignDataRetrievingFunction(self.latest_tracking_window.returnTrackedSignals)  # type: ignore[attr-defined]
            _safe_call_update(window, ("update_newParams",), params)
        elif name == "Vitals":
            if self.latest_tracking_window is not None:
                window.assignDataRetrievingFunction(self.latest_tracking_window.returnTrackedSignals)  # type: ignore[attr-defined]
            _safe_call_update(window, ("update_newParams_andHR",), params, self.hrs)

    def _new_phase_window(self) -> QWidget:
        window = PhaseUnWr_plot.PlotWindow()
        self._update_normal_window(window, "Phase Unwrap")
        return window

    def _new_stft_window(self) -> QWidget:
        window = HR_STFT_plot.PlotWindow()
        self._update_normal_window(window, "HR STFT")
        return window

    def _new_doa2d_window(self) -> QWidget:
        window = DoA_2DAziPlane.PlotWindow()
        self._update_normal_window(window, "DoA 2D Azi Plane")
        return window

    def _new_tracking_window(self) -> QWidget:
        window = DoA_2DTracking.PlotWindow()
        self._update_normal_window(window, "Tracking")
        self.latest_tracking_window = window
        return window

    def _ensure_tracking_for_dependents(self) -> None:
        if self.latest_tracking_window is None or self.latest_tracking_window.isHidden():
            self.launch_normal_window("Tracking")

    def _new_tracked_signals_window(self) -> QWidget:
        self._ensure_tracking_for_dependents()
        window = TrackingPlot.PlotWindow()
        self._update_normal_window(window, "Tracked Signals")
        return window

    def _new_vitals_window(self) -> QWidget:
        self._ensure_tracking_for_dependents()
        window = VitalsPlot.PlotWindow()
        self._update_normal_window(window, "Vitals")
        return window

    # ------------------------------------------------------------------
    # Autorun
    # ------------------------------------------------------------------
    def _autorun_if_enabled(self) -> None:
        if not AUTORUN.get("enabled", False):
            return
        if AUTORUN.get("process_normal", False):
            self.process_normal_data()
        if AUTORUN.get("process_mesh", False):
            self.process_mesh_data()
        for name in AUTORUN.get("launch_normal_windows", []):
            if name == "Vitals":
                self.launch_vitals_from_current_tracking()
            elif name in self._normal_specs():
                self.launch_normal_window(name)
        if AUTORUN.get("launch_mesh_window", False):
            self.launch_mesh_window()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
