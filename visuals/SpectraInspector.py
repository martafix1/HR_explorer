# from __future__ import annotations

import os
import sys
from typing import Any, Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QSplitter, QVBoxLayout, QWidget
import matplotlib.pyplot as plt
from PySide6.QtGui import QColor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import processing.slidingFFT as slidingFFT
import visuals.param_controls as pctrl


class SpectraInspectorWindow(QWidget):
    """Interactive FFT spectrum inspector for one tracked signal."""

    def __init__(self):
        super().__init__()
        self.initDone = False
        self.setWindowTitle("Spectra inspector")
        self.resize(1080, 720)

        self.tracking_data: dict[int, dict[str, np.ndarray]] = {}
        self.params: dict[str, Any] = {"frame_index2time": 0.05}
        self.HRs: dict[str, dict[str, np.ndarray]] = {}
        self.vital_processing_values: dict[str, Any] = {}
        self.vital_signal_processor: Callable[..., np.ndarray] | None = None
        self.enabled_track_ids: list[int] = []
        self._snapping_frame = False
        self._previous_inspector_methods: list[str] = []

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle { background-color: #404040; }
            QSplitter::handle:hover { background-color: #606060; }
        """)

        ctrl_widget = QWidget()
        grid = QGridLayout(ctrl_widget)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)
        ctrl_panel = pctrl.ControlPanel(grid, callback=self.update_onSliderMove)

        self.frame_ctrl = pctrl.SliderControl("Frame", min_val=0, max_val=200, default=40)
        ctrl_panel.add(self.frame_ctrl, row=0, col=0, col_span=2)

        # self.track_ctrl = pctrl.DictTableControl(
        #     "Track",
        #     params={"Track ID": ["0"]},
        #     orientation="horizontal",
        # )
        # ctrl_panel.add(self.track_ctrl, row=0, col=2)

        self.track_ctrl = pctrl.ComboControl(
                "Track ID",
                options=["Load data"],
                default=0,
            )
        ctrl_panel.add(self.track_ctrl, row=0, col=2)

        self.freq_range_ctrl = pctrl.RangeControl(
            "Shown freq range",
            min_val=0,
            max_val=5000,
            default=(0, 2500),
            unit="Hz",
            conv=lambda x: x / 1000.0,
        )
        ctrl_panel.add(self.freq_range_ctrl, row=1, col=0, col_span=2)

        self.time_content_ctrl = pctrl.ChecksetControl(
            "Time plot",
            labels=["Show input", "Show FFT frequency"],
            defaults=[True, True],
        )
        ctrl_panel.add(self.time_content_ctrl, row=1, col=2)

        self.ref_signal_ctrl = pctrl.DictTableControl_composite(
            "Reference signal plotting",
            params={},
            headers=["TIME", "SPECTRUM"],
            orientation="vertical",
        )
        ctrl_panel.add(self.ref_signal_ctrl, row=0, col=3, row_span=2)

        method_params = self.buildMethodControlDict()
        self.method_ctrl = pctrl.DictTableControl_composite(
            "FFT methods",
            params=method_params,
            headers=["SHOW", "METHOD", "INPUT", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6", "Param7", "Param8"],
            orientation="horizontal",
        )
        self._previous_inspector_methods = [
            row["Method"] for row in self.method_ctrl.value().values()
        ]
        ctrl_panel.add(self.method_ctrl, row=2, col=0, col_span=4, row_span=3)

        ctrl_widget.setMaximumHeight(360)
        splitter.addWidget(ctrl_widget)

        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        plot_splitter.setHandleWidth(6)
        self.time_plot = pg.PlotWidget()
        self.time_plot.setBackground("w")
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.time_legend = self.time_plot.addLegend()
        self.time_plot.setLabel("bottom", "Frame")

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("w")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_legend = self.spectrum_plot.addLegend()
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")

        plot_splitter.addWidget(self.time_plot)
        plot_splitter.addWidget(self.spectrum_plot)
        splitter.addWidget(plot_splitter)
        splitter.setSizes([260, 460])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self.initDone = True
        self.ref_cmap = plt.get_cmap('tab10')
        self.main_cmap = plt.get_cmap('Set1')
        self.update_onSliderMove()

    def fftCapabilities(self) -> dict[str, dict[str, Any]]:
        return {
            "slidingFFT": {
                "initFrames": 20,
                "stepFrames": 10,
                "sig_sample": 80,
                "fixedFFT_size": 200,
                "freqRangeStart": 0.1,
                "freqRangeStop": 2.0,
                "parabolicInterpolation": False,
                "window": ["rect", "hann", "hamming","blackman"],
            }
        }

    def buildMethodControlDict(self, control_values: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        caps = self.fftCapabilities()
        inputs = self._input_options()
        output = {}

        for i in range(3):
            name = f"FFT {i + 1}"
            prev = control_values.get(name, {}) if control_values else {}
            selected = prev.get("Method", "slidingFFT" if i == 0 else "None")
            if selected not in caps and selected != "None":
                selected = "None"

            methods = [selected] + ["None"] + list(caps.keys())
            input_name = prev.get("Input", inputs[0])
            row = {
                "Show": prev.get("Show", i == 0),
                "Method": list(dict.fromkeys(methods)),
                "Input": list(dict.fromkeys([input_name] + inputs)),
            }

            if selected in caps:
                for param_name, default in caps[selected].items():
                    if isinstance(default, list):
                        row[param_name] = default
                    else:
                        row[param_name] = prev.get(param_name, default)
            output[name] = row
        return output

    def update_context(
        self,
        tracking_data: dict[int, dict[str, np.ndarray]] | None,
        params: dict[str, Any] | None,
        HRs: dict[str, dict[str, np.ndarray]] | None = None,
        vital_processing_values: dict[str, Any] | None = None,
        vital_signal_processor: Callable[..., np.ndarray] | None = None,
        enabled_track_ids: list[int] | None = None,
    ) -> None:
        """Receive current VitalExtraction state."""
        self.initDone = False
        self.tracking_data = tracking_data or {}
        self.params = params or {"frame_index2time": 0.05}
        self.HRs = HRs or {}
        self.vital_processing_values = vital_processing_values or {}
        self.vital_signal_processor = vital_signal_processor
        self.enabled_track_ids = enabled_track_ids or sorted(self.tracking_data.keys())

        self._refresh_track_control()
        self._refresh_ref_control()
        self._refresh_frame_control()
        self.method_ctrl.set_newParams(self.buildMethodControlDict(self.method_ctrl.value()), silent=True)
        self._previous_inspector_methods = [row["Method"] for row in self.method_ctrl.value().values()]
        self.initDone = True
        self.update_onSliderMove()

    def _refresh_track_control(self) -> None:
        ids = [str(i) for i in self.enabled_track_ids if i in self.tracking_data]
        if not ids:
            ids = [str(i) for i in sorted(self.tracking_data.keys())] or ["0"]
        
        

        current = str(self.track_ctrl.value())
        options = [current] + [item for item in ids if item != current]
        self.track_ctrl.update_options(options)

    def _refresh_ref_control(self) -> None:
        ref_params = {}
        for ref_name in self.HRs.keys():
            ref_params[ref_name] = {"Time": False, "Spectrum": False}
        self.ref_signal_ctrl.set_newParams(ref_params, headers=["TIME", "SPECTRUM"], silent=True)

    def _refresh_frame_control(self) -> None:
        track_id = self._selected_track_id()
        frames = self._track_frames(track_id)
        if len(frames) == 0:
            self.frame_ctrl.set_range(0, 1)
            return
        lo, hi = int(np.min(frames)), int(np.max(frames))
        self.frame_ctrl.set_range(lo, hi)

    def _input_options(self) -> list[str]:
        inputs = ["Ph. unwrap"]
        for line_name, line_vals in self.vital_processing_values.items():
            if line_vals.get("Method") != "None":
                inputs.append(line_name)
        return inputs

    def _selected_track_id(self) -> int:
        try:
            return int(self.track_ctrl.value())
        except (TypeError, ValueError):
            return 0

    def _track_frames(self, track_id: int) -> np.ndarray:
        if track_id not in self.tracking_data:
            return np.array([], dtype=int)
        return np.asarray(self.tracking_data[track_id].get("frames", np.array([], dtype=int)))

    def _common_fft_timing(self, method_rows: dict[str, Any]) -> tuple[int, int]:
        init_frames = [int(row.get("initFrames", 0)) for row in method_rows.values() if row.get("Show") and row.get("Method") != "None"]
        step_frames = [int(row.get("stepFrames", 1)) for row in method_rows.values() if row.get("Show") and row.get("Method") != "None"]
        return (max(init_frames) if init_frames else 0, max(step_frames) if step_frames else 1)

    def _frame_to_sample_index(self, frames: np.ndarray, frame_value: int) -> int:
        if len(frames) == 0:
            return 0
        return int(np.argmin(np.abs(frames - frame_value)))

    def _sample_index_to_frame(self, frames: np.ndarray, sample_idx: int) -> int:
        if len(frames) == 0:
            return 0
        sample_idx = int(np.clip(sample_idx, 0, len(frames) - 1))
        return int(frames[sample_idx])

    def _selected_frame_and_sample(self, method_rows: dict[str, Any]) -> tuple[int, int]:
        raw_frame = int(self.frame_ctrl.value())
        frames = self._track_frames(self._selected_track_id())
        raw_sample = self._frame_to_sample_index(frames, raw_frame)
        init_frame, step_frame = self._common_fft_timing(method_rows)
        step_frame = max(step_frame, 1)
        if raw_sample < init_frame:
            snapped_sample = init_frame
        else:
            snapped_sample = init_frame + round((raw_sample - init_frame) / step_frame) * step_frame
        if len(frames):
            snapped_sample = int(np.clip(snapped_sample, 0, len(frames) - 1))
        snapped_frame = self._sample_index_to_frame(frames, snapped_sample)
        if snapped_frame != raw_frame and not self._snapping_frame:
            self._snapping_frame = True
            self.frame_ctrl.set_value(snapped_frame)
            self._snapping_frame = False
        return snapped_frame, snapped_sample

    def _source_signals(self, track_id: int) -> dict[str, np.ndarray]:
        if track_id not in self.tracking_data:
            return {}
        base = np.asarray(self.tracking_data[track_id].get("phase_unwrapped", np.array([])), dtype=float)
        signals: dict[str, np.ndarray] = {"Ph. unwrap": base}
        if self.vital_signal_processor is None:
            return signals

        for line_name, line_vals in self.vital_processing_values.items():
            method = line_vals.get("Method", "None")
            if method == "None":
                continue
            source_name = line_vals.get("Input", "Ph. unwrap")
            if source_name not in signals or len(signals[source_name]) == 0:
                continue
            try:
                signals[line_name] = self.vital_signal_processor(
                    signal=signals[source_name],
                    method=method,
                    m_params=line_vals,
                )
            except Exception as exc:
                print(f"SpectraInspector skipped {line_name}: {exc}")
        return signals

    def _maybe_rebuild_method_table(self) -> dict[str, Any]:
        current = self.method_ctrl.value()
        selected_methods = [row.get("Method", "None") for row in current.values()]
        if selected_methods != self._previous_inspector_methods:
            self.method_ctrl.set_newParams(self.buildMethodControlDict(current), silent=True)
            self._previous_inspector_methods = selected_methods
            current = self.method_ctrl.value()
        return current

    def update_onSliderMove(self) -> None:
        if not self.initDone:
            return
        if self._snapping_frame:
            return

        method_rows = self._maybe_rebuild_method_table()
        self._refresh_frame_control()

        track_id = self._selected_track_id()
        frames = self._track_frames(track_id)
        source_signals = self._source_signals(track_id)
        frame_idx, sample_idx = self._selected_frame_and_sample(method_rows)
        fs = 1 / float(self.params.get("frame_index2time", 0.05))
        f_lo, f_hi = self.freq_range_ctrl.value()
        shown_freq_range = (f_lo / 1000.0, f_hi / 1000.0)

        self.time_plot.clear()
        self.spectrum_plot.clear()

        self.timePlotMaxVal = 0 # for drawing lines on top without infi scaling the plot

        if len(frames) == 0 or not source_signals:
            return

        show_input, show_fft_freq = self.time_content_ctrl.value()
        

        current_line = pg.InfiniteLine(frame_idx, angle=90, pen=pg.mkPen("k", width=2))
        self.time_plot.addItem(current_line)

        already_showing_input = []

        for idx, (method_name, row) in enumerate(method_rows.items()):
            if not row.get("Show") or row.get("Method") == "None":
                continue
            source_name = row.get("Input", "Ph. unwrap")
            if source_name not in source_signals:
                continue
            signal = np.asarray(source_signals[source_name], dtype=float)
            rgba = self.main_cmap(idx % self.main_cmap.N)
            color = QColor.fromRgbF(*rgba)
            pg.intColor(idx, hues=8)
            color_darker = color.darker(140)
            pen = pg.mkPen(color=color, width=2)

            inst_f, _inst_a = slidingFFT.instantaneous_analysis_FFT(
                signal,
                float(fs),
                int(row["initFrames"]),
                int(row["stepFrames"]),
                int(row["sig_sample"]),
                int(row["fixedFFT_size"]),
                float(row["freqRangeStart"]),
                float(row["freqRangeStop"]),
                bool(row["parabolicInterpolation"]),
                str(row["window"]),
                # inspectFrame=sample_idx,
                # returnSpectrum=True,
            )

            answer = slidingFFT.FFT_single(signal,sample_idx,
                float(fs),
                int(row["initFrames"]),
                int(row["stepFrames"]),
                int(row["sig_sample"]),
                int(row["fixedFFT_size"]),
                float(row["freqRangeStart"]),
                float(row["freqRangeStop"]),
                bool(row["parabolicInterpolation"]),
                str(row["window"])
                )
            if answer is None:
                continue
            else:
                peak_freq, peak_amp, amplSpect = answer

            backStop = max(0,sample_idx - int(row["sig_sample"]))
            if show_fft_freq:
                self.time_plot.plot(frames[: len(inst_f)], inst_f[: len(frames)], pen=pen, name=f"{method_name} f")
                self.timePlotMaxVal = max(np.max(inst_f[: len(frames)]),self.timePlotMaxVal)
                back_frame = self._sample_index_to_frame(frames, backStop)
                self._plot_fft_extent(frame_idx, back_frame, color)

            if show_input and source_name not in already_showing_input:
                self.time_plot.plot(
                    frames,
                    source_signals[source_name],
                    pen=pg.mkPen(color=color_darker, width=1),
                    name=f"ID:{track_id} {source_name}",
                )
                already_showing_input.append(source_name)

                mbyMax = np.max(source_signals[source_name])
                self.timePlotMaxVal = max(mbyMax,self.timePlotMaxVal)

            freqs = np.arange(int(row["fixedFFT_size"])//2 + 1) * fs / int(row["fixedFFT_size"])
            amplitude = amplSpect
            mask = (freqs >= shown_freq_range[0]) & (freqs <= shown_freq_range[1])
            self.spectrum_plot.plot(freqs[mask], amplitude[mask], pen=pen, name=f"{method_name} spectrum")

            peak_pen = pg.mkPen(color=color, width=1, style=Qt.DashLine)
            self.spectrum_plot.addItem(pg.InfiniteLine(peak_freq, angle=90, pen=peak_pen))
            self.spectrum_plot.plot(
                [peak_freq],
                [peak_amp],
                pen=None,
                symbol="o",
                symbolPen=peak_pen,
                symbolBrush=color,
                symbolSize=8,
                name=f"{method_name} peak {peak_freq:.3g} Hz",
            )
            self.spectrum_plot.addItem(pg.InfiniteLine(peak_amp, angle=0, pen=peak_pen))

        self._plot_reference_signals(frame_idx)
        self.spectrum_plot.setXRange(*shown_freq_range, padding=0)

    def _plot_fft_extent(self, frame_idx: int, back_idx: int, color: Any) -> None:
        back_pen = pg.mkPen(color=color, width=1, style=Qt.DashLine)
        self.time_plot.addItem(pg.InfiniteLine(back_idx, angle=90, pen=back_pen))
        y_min, y_max = self.time_plot.viewRange()[1]
        y = self.timePlotMaxVal * (0.95)
        self.time_plot.plot([back_idx, frame_idx], [y, y], pen=back_pen)
        

    def _plot_reference_signals(self, frame_idx: int) -> None:
        ref_selection = self.ref_signal_ctrl.value()
        for idx, (ref_name, flags) in enumerate(ref_selection.items()):
            if ref_name not in self.HRs:
                continue
            frames = np.asarray(self.HRs[ref_name].get("frames", np.array([])))
            values = np.asarray(self.HRs[ref_name].get("hr", np.array([])))
            if len(frames) == 0 or len(values) == 0:
                continue
            rcolor, g, b, acolor = self.ref_cmap(idx % 10)
            color = (int(rcolor * 255), int(g * 255), int(b * 255), int(acolor * 255))
            pen = pg.mkPen(color=color, width=2, style=Qt.DashDotLine)
            if flags.get("Time"):
                self.time_plot.plot(frames, values, pen=pen, name=f"REF {ref_name}")
                self.timePlotMaxVal = max(np.max(values),self.timePlotMaxVal)
            if flags.get("Spectrum"):
                nearest = int(np.argmin(np.abs(frames - frame_idx)))
                ref_freq = float(values[nearest])
                self.spectrum_plot.addItem(pg.InfiniteLine(ref_freq, angle=90, pen=pen))


def dummy_tracking_data() -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    frames = np.arange(0, 320)
    fs = 20.0
    t = frames / fs
    freq = 0.75 + 0.08 * np.sin(2 * np.pi * 0.02 * t)
    phase = np.cumsum(2 * np.pi * freq / fs)
    signal = np.sin(phase) + 0.2 * np.sin(2 * np.pi * 1.6 * t)
    tracking = {0: {"frames": frames, "phase_unwrapped": signal}}
    params = {"frame_index2time": 1 / fs, "i_Frames_begin": 0, "i_Frames_end": len(frames)}
    refs = {"dummy_hr": {"frames": frames, "hr": freq}}
    return tracking, params, refs


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpectraInspectorWindow()
    tracking, params, refs = dummy_tracking_data()
    window.update_context(tracking, params, refs)
    window.show()
    sys.exit(app.exec())
