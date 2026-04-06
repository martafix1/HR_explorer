"""
param_controls.py
=================

Two public classes:

    ControlPanel  — context object you create once per window.
                    Holds the target QGridLayout and the single callback.
                    Use .add() to place controls.

    ParamControl  — base class (QGroupBox). Never instantiate directly.
        SliderControl   — integer slider + spinbox
        RangeControl    — dual spinbox + range slider
        ComboControl    — drop-down menu
        ChecksetControl — row of checkboxes
        ButtonControl   — push button

Title format (set automatically on every value change):
    "Name  [value_str | Δ delta_str unit]"

Flash cue: the GroupBox border flashes for FLASH_DURATION_MS when the
value changes.  Gate the whole feature with FLASH_ENABLED = True/False.
"""

from __future__ import annotations

import sys
from typing import Callable, Any

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QSplitter,
    QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSlider, QSpinBox, QComboBox, QCheckBox, QPushButton,
    QLabel, QSizePolicy, 
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont

try:
    from superqt import QRangeSlider
    _HAS_SUPERQT = True
except ImportError:
    _HAS_SUPERQT = False

import pyqtgraph as pg


# ---------------------------------------------------------------------------
# Global feature flag — set False to silence all flash cues
# ---------------------------------------------------------------------------
FLASH_ENABLED: bool = True
FLASH_DURATION_MS: int = 50

# Flash colours (border only — works in both light and dark Qt themes)
_FLASH_STYLE  = "QGroupBox { border: 2px solid #3ddc84; border-radius: 4px; }"
_NORMAL_STYLE = """
QGroupBox {
    border: 2px solid palette(mid);
    border-radius: 4px;
}
QGroupBox::title {
    color: palette(text);
}
"""

# extraordinary states warning:

_STATE_NORMAL  = 0
_STATE_WARNING = 1
_STATE_ERROR   = 2

_WARN_STYLE = """
QGroupBox {
    border: 2px solid #cc7a00;
    border-radius: 4px;
}
QGroupBox::title {
    color: #cc7a00;
    font-weight: bold;
}
"""

_ERR_STYLE = """
QGroupBox {
    border: 2px solid #cc0000;
    border-radius: 4px;
}
QGroupBox::title {
    color: #cc0000;
    font-weight: bold;
}
"""

# ---------------------------------------------------------------------------
# Scroll-wheel filter — prevents accidental slider nudges while scrolling
# ---------------------------------------------------------------------------
from PySide6.QtCore import QObject, QEvent

class _WheelFilter(QObject):
    def eventFilter(self, obj, event):                       # type: ignore[override]
        if event.type() == QEvent.Type.Wheel:
            return True          # eat the event
        return super().eventFilter(obj, event)

import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import visuals.utils as utils
_WHEEL_FILTER = utils.SliderWheelFilter()


def _wheel_filter() -> _WheelFilter:
    global _WHEEL_FILTER
    if _WHEEL_FILTER is None:
        _WHEEL_FILTER = _WheelFilter()
    return _WHEEL_FILTER


# ---------------------------------------------------------------------------
# ControlPanel — factory / context object
# ---------------------------------------------------------------------------

class ControlPanel:
    """
    Create one of these per window before building controls.

    Usage::

        panel = ControlPanel(grid_layout, callback=self.on_change)
        panel.add(self.freq_slider,  row=0, col=0)
        panel.add(self.frame_range,  row=0, col=1, col_span=2)

    The callback signature is ``callback() -> None``.
    It is called once after any control settles its internal sync.
    """

    def __init__(self, grid: QGridLayout, callback: Callable[[], None]):
        self._grid     = grid
        self._callback = callback
        # keep equal stretch on all columns so controls fill space
        for c in range(grid.columnCount() or 3):
            grid.setColumnStretch(c, 1)

    def add(
        self,
        control: "ParamControl",
        row: int,
        col: int,
        col_span: int = 1,
    ) -> None:
        """Place *control* in the grid and wire its callback."""
        # ensure column stretches cover new columns
        for c in range(col, col + col_span):
            if self._grid.columnStretch(c) == 0:
                self._grid.setColumnStretch(c, 1)

        self._grid.addWidget(control, row, col, 1, col_span)
        control._set_callback(self._callback)


# ---------------------------------------------------------------------------
# ParamControl — base class
# ---------------------------------------------------------------------------

class ParamControl(QGroupBox):
    """
    Base for all parameter controls.

    Subclasses must implement:
        _build_interior()   — populate self._inner_layout
        value()             — return current value
        set_value(v)        — programmatic set (suppressed, no callback)
        set_range(lo, hi)   — update limits, ratio-preserving (suppressed)
        _value_str()        — short string for the title  e.g. "12"
    """

    value_changed = Signal()   # emitted once after internal sync settles

    # ---- sizing knobs (override per subclass if needed) ----
    _MAX_HEIGHT: int = 68

    def __init__(self, name: str, unit: str = "", conv: Callable[[int], float] | None = None):
        super().__init__()
        self._name      = name
        self._unit      = unit
        self._conv      = conv          # index → physical value (for display only)
        self._callback: Callable[[], None] | None = None
        self._syncing   = False
        self._delta: float | None = None   # set from outside via set_delta()

        self._state = _STATE_NORMAL
        self._state_msg: str | None = None
        self._error_lock_value: Any = None

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._flash_off)

        font = QFont()
        font.setFamilies(["Segoe UI Symbol", "DejaVu Sans", "Sans Serif"])
        self.setFont(font)

        self.setStyleSheet(_NORMAL_STYLE)
        self.setMaximumHeight(self._MAX_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 14, 4, 4)
        outer.setSpacing(2)
        self._inner_layout = QHBoxLayout()
        self._inner_layout.setSpacing(4)
        outer.addLayout(self._inner_layout)

        self._build_interior()
        self._refresh_title()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, v: Any) -> None:
        raise NotImplementedError

    def set_range(self, lo: int, hi: int) -> None:
        """Override in slider-type subclasses. Default is no-op."""

    def set_delta(self, delta: float) -> None:
        """Call this from your update callback to show Δ in the title."""
        self._delta = delta
        self._refresh_title()

    def set_conv(self, conv: Callable[[int], float] | None = None) -> None:
        self._conv = conv
        self._refresh_title()

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------

    def set_warning(self, msg: str) -> None:
        self._state = _STATE_WARNING
        self._state_msg = msg
        self._apply_state_style()
        self._refresh_title()

    def set_error(self, msg: str, lock_value: Any | None = None) -> None:
        self._state = _STATE_ERROR
        self._state_msg = msg
        self._error_lock_value = lock_value

        if lock_value is not None:
            self.set_value(lock_value)

        self._set_enabled_recursive(False)

        self._apply_state_style()
        self._refresh_title()

    def clear_state(self) -> None:
        self._state = _STATE_NORMAL
        self._state_msg = None
        self._error_lock_value = None

        self._set_enabled_recursive(True)

        self.setStyleSheet(_NORMAL_STYLE)
        self._refresh_title()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_callback(self, cb: Callable[[], None]) -> None:
        self._callback = cb

    def _emit(self) -> None:
        """Flash, refresh title, then fire the external callback once."""
        if self._state == _STATE_ERROR:
            return
        self._refresh_title()
        self._flash_on()
        self.value_changed.emit()
        if self._callback:
            self._callback()

    # this is to dissallow value changes in error state
    def _set_enabled_recursive(self, enabled: bool) -> None:
        for child in self.findChildren(QWidget):
            child.setEnabled(enabled)

    def _refresh_title(self) -> None:
        parts = [self._value_str()]
        if self._delta is not None:
            d_str = f"Δ {self._delta:.3g}"
            if self._unit:
                d_str += f" {self._unit}"
            parts.append(d_str)
        elif self._unit:
            parts.append(self._unit)
        inner = "  |  ".join(parts)

        title = f"{self._name}  [{inner}]"

        # append state message
        if self._state == _STATE_WARNING and self._state_msg:
            title += f"  ⚠ warn: {self._state_msg}"
        elif self._state == _STATE_ERROR and self._state_msg:
            title += f"  ⛒ err: {self._state_msg}"

        self.setTitle(title)

    def _value_str(self) -> str:
        return str(self.value())

    def _flash_on(self) -> None:
        if not FLASH_ENABLED:
            return
        self.setStyleSheet(_FLASH_STYLE)
        self._flash_timer.start(FLASH_DURATION_MS)

    def _flash_off(self) -> None:
        self._apply_state_style()

    def _build_interior(self) -> None:
        raise NotImplementedError

    def _apply_state_style(self) -> None:
        if self._state == _STATE_WARNING:
            self.setStyleSheet(_WARN_STYLE)
        elif self._state == _STATE_ERROR:
            self.setStyleSheet(_ERR_STYLE)
        else:
            self.setStyleSheet(_NORMAL_STYLE)

    # ------------------------------------------------------------------
    # Ratio-preserving range helper (shared by slider types)
    # ------------------------------------------------------------------

    @staticmethod
    def _ratio_value(old_val: int, old_lo: int, old_hi: int,
                     new_lo: int, new_hi: int) -> int:
        """Return the value that preserves the 0-1 ratio after a range change."""
        old_span = old_hi - old_lo
        new_span = new_hi - new_lo
        if old_span == 0:
            return new_lo
        ratio = (old_val - old_lo) / old_span
        return round(new_lo + ratio * new_span)


# ---------------------------------------------------------------------------
# SliderControl
# ---------------------------------------------------------------------------

class SliderControl(ParamControl):
    """
    Integer slider with a spinbox for direct input.

    Layout:  [spinbox]  [=========slider=========]
    Title:   "Name  [idx: 12 | Δ 0.05 m]"
    """

    _MAX_HEIGHT = 64

    def __init__(
        self,
        name: str,
        min_val: int = 0,
        max_val: int = 100,
        default: int = 0,
        unit: str = "",
        conv: Callable[[int], float] | None = None,
    ):
        self._min = min_val
        self._max = max_val
        self._default = default
        super().__init__(name, unit, conv)
        # set_value after widgets exist
        self._silent_set(default)

    def _build_interior(self) -> None:
        self._spin = QSpinBox()
        self._spin.setRange(self._min, self._max)
        self._spin.setFixedWidth(58)
        self._spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._spin.installEventFilter(_wheel_filter())

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(self._min, self._max)
        self._slider.installEventFilter(_wheel_filter())

        self._inner_layout.addWidget(self._spin)
        self._inner_layout.addWidget(self._slider, stretch=1)

        # internal sync (no external callback)
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    # ---- internal sync ----

    def _on_slider(self, v: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self._spin.setValue(v)
        self._syncing = False
        self._emit()

    def _on_spin(self, v: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self._slider.setValue(v)
        self._syncing = False
        self._emit()

    def _silent_set(self, v: int) -> None:
        self._syncing = True
        self._slider.setValue(v)
        self._spin.setValue(v)
        self._syncing = False
        self._refresh_title()

    # ---- public API ----

    def value(self) -> int:
        return self._slider.value()

    def set_value(self, v: int) -> None:
        self._silent_set(v)

    def set_range(self, lo: int, hi: int) -> None:
        old_val = self.value()
        new_val = self._ratio_value(old_val, self._min, self._max, lo, hi)
        new_val = max(lo, min(hi, new_val))
        self._min, self._max = lo, hi
        self._syncing = True
        self._slider.setRange(lo, hi)
        self._spin.setRange(lo, hi)
        self._slider.setValue(new_val)
        self._spin.setValue(new_val)
        self._syncing = False
        self._refresh_title()

    def _value_str(self) -> str:
        v = self.value()
        if self._conv:
            return f"idx {v}  ~  {self._conv(v):.3g}"
        return f"idx {v}"


# ---------------------------------------------------------------------------
# RangeControl
# ---------------------------------------------------------------------------

class RangeControl(ParamControl):
    """
    Dual-handle range slider with spinboxes on each side.

    Layout:  [spin_lo]  [=====|range|=====]  [spin_hi]
    value() returns (lo, hi) as a tuple of ints.
    """

    _MAX_HEIGHT = 64

    def __init__(
        self,
        name: str,
        min_val: int = 0,
        max_val: int = 100,
        default: tuple[int, int] | None = None,
        unit: str = "",
        conv: Callable[[int], float] | None = None,
    ):
        if not _HAS_SUPERQT:
            raise ImportError("RangeControl requires the 'superqt' package.")
        self._min = min_val
        self._max = max_val
        self._default = default or (min_val, max_val)
        super().__init__(name, unit, conv)
        self._silent_set(self._default)

    def _build_interior(self) -> None:
        self._spin_lo = QSpinBox()
        self._spin_lo.setRange(self._min, self._max)
        self._spin_lo.setFixedWidth(58)
        self._spin_lo.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._spin_lo.installEventFilter(_wheel_filter())

        self._spin_hi = QSpinBox()
        self._spin_hi.setRange(self._min, self._max)
        self._spin_hi.setFixedWidth(58)
        self._spin_hi.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._spin_hi.installEventFilter(_wheel_filter())

        self._rslider = utils.HighlightRangeSlider(Qt.Orientation.Horizontal)
        self._rslider.setRange(self._min, self._max)
        self._rslider.setStyleSheet("QRangeSlider { qproperty-barColor: #3ddc84; }")
        self._rslider.installEventFilter(_wheel_filter())

        self._inner_layout.addWidget(self._spin_lo)
        self._inner_layout.addWidget(self._rslider, stretch=1)
        self._inner_layout.addWidget(self._spin_hi)

        self._rslider.valueChanged.connect(self._on_rslider)
        self._spin_lo.valueChanged.connect(self._on_spin_lo)
        self._spin_hi.valueChanged.connect(self._on_spin_hi)

    # ---- internal sync ----

    def _on_rslider(self, vals) -> None:
        if self._syncing:
            return
        lo, hi = vals
        self._syncing = True
        self._spin_lo.setValue(lo)
        self._spin_hi.setValue(hi)
        self._syncing = False
        self._emit()

    def _on_spin_lo(self, v: int) -> None:
        if self._syncing:
            return
        hi = self._rslider.value()[1]
        if v > hi:
            v = hi
        self._syncing = True
        self._rslider.setValue((v, hi))
        self._syncing = False
        self._emit()

    def _on_spin_hi(self, v: int) -> None:
        if self._syncing:
            return
        lo = self._rslider.value()[0]
        if v < lo:
            v = lo
        self._syncing = True
        self._rslider.setValue((lo, v))
        self._syncing = False
        self._emit()

    def _silent_set(self, v: tuple[int, int]) -> None:
        lo, hi = v
        self._syncing = True
        self._rslider.setValue((lo, hi))
        self._spin_lo.setValue(lo)
        self._spin_hi.setValue(hi)
        self._syncing = False
        self._refresh_title()

    # ---- public API ----

    def value(self) -> tuple[int, int]:
        lo, hi = self._rslider.value()
        return (int(lo), int(hi))

    def set_value(self, v: tuple[int, int]) -> None:
        self._silent_set(v)

    def set_range(self, lo: int, hi: int) -> None:
        old_lo, old_hi = self.value()
        new_lo = self._ratio_value(old_lo, self._min, self._max, lo, hi)
        new_hi = self._ratio_value(old_hi, self._min, self._max, lo, hi)
        new_lo = max(lo, min(hi, new_lo))
        new_hi = max(lo, min(hi, new_hi))
        if new_lo > new_hi:
            new_lo = new_hi
        self._min, self._max = lo, hi
        self._syncing = True
        self._rslider.setRange(lo, hi)
        self._spin_lo.setRange(lo, hi)
        self._spin_hi.setRange(lo, hi)
        self._rslider.setValue((new_lo, new_hi))
        self._spin_lo.setValue(new_lo)
        self._spin_hi.setValue(new_hi)
        self._syncing = False
        self._refresh_title()

    def _value_str(self) -> str:
        lo, hi = self.value()
        if self._conv:
            return f"idx {lo}-{hi}  ~  {self._conv(lo):.3g} – {self._conv(hi):.3g}"
        return f"idx {lo} – {hi}"


# ---------------------------------------------------------------------------
# ComboControl
# ---------------------------------------------------------------------------

class ComboControl(ParamControl):
    """Drop-down selection. value() returns the selected string."""

    _MAX_HEIGHT = 52

    def __init__(self, name: str, options: list[str], default: int = 0):
        self._options = options
        self._default_idx = default
        super().__init__(name)
        self._silent_set(default)

    def _build_interior(self) -> None:
        self._combo = QComboBox()
        self._combo.addItems(self._options)
        self._combo.installEventFilter(_wheel_filter())
        self._inner_layout.addWidget(self._combo, stretch=1)
        self._combo.currentIndexChanged.connect(self._on_change)

    def _on_change(self, _idx: int) -> None:
        if self._syncing:
            return
        self._emit()

    def _silent_set(self, idx: int) -> None:
        self._syncing = True
        self._combo.setCurrentIndex(idx)
        self._syncing = False
        self._refresh_title()

    def value(self) -> str:
        return self._combo.currentText()

    def set_value(self, v: str | int) -> None:
        self._syncing = True
        if isinstance(v, int):
            self._combo.setCurrentIndex(v)
        else:
            idx = self._combo.findText(v)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
        self._syncing = False
        self._refresh_title()

    def _value_str(self) -> str:
        return self.value()


# ---------------------------------------------------------------------------
# ChecksetControl
# ---------------------------------------------------------------------------

class ChecksetControl(ParamControl):
    """
    Row of labelled checkboxes.
    value() returns list[bool], one entry per checkbox in order.
    """

    _MAX_HEIGHT = 52

    def __init__(self, name: str, labels: list[str], defaults: list[bool] | None = None):
        self._labels   = labels
        self._defaults = defaults or [False] * len(labels)
        super().__init__(name)
        self._silent_set(self._defaults)

    def _build_interior(self) -> None:
        self._boxes: list[QCheckBox] = []
        for lbl in self._labels:
            cb = QCheckBox(lbl)
            cb.stateChanged.connect(self._on_change)
            self._inner_layout.addWidget(cb)
            self._boxes.append(cb)
        self._inner_layout.addStretch(1)

    def _on_change(self, _state: int) -> None:
        if self._syncing:
            return
        self._emit()

    def _silent_set(self, vals: list[bool]) -> None:
        self._syncing = True
        for cb, v in zip(self._boxes, vals):
            cb.setChecked(v)
        self._syncing = False
        self._refresh_title()

    def value(self) -> list[bool]:
        return [cb.isChecked() for cb in self._boxes]

    def set_value(self, vals: list[bool]) -> None:
        self._silent_set(vals)

    def _value_str(self) -> str:
        checked = [l for l, cb in zip(self._labels, self._boxes) if cb.isChecked()]
        return ", ".join(checked) if checked else "none"


# ---------------------------------------------------------------------------
# ButtonControl
# ---------------------------------------------------------------------------

class ButtonControl(ParamControl):
    """
    A single push button.
    'value()' returns True momentarily (bool); in practice use the callback.
    The button label is separate from the GroupBox name.
    """

    _MAX_HEIGHT = 52

    def __init__(self, name: str, button_label: str = "Go"):
        self._button_label = button_label
        self._pressed      = False
        super().__init__(name)

    def _build_interior(self) -> None:
        self._btn = QPushButton(self._button_label)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._inner_layout.addWidget(self._btn)
        self._btn.clicked.connect(self._on_click)

    def _on_click(self) -> None:
        self._pressed = True
        self._emit()
        self._pressed = False
        self._refresh_title()

    def value(self) -> bool:
        return self._pressed

    def set_value(self, v: Any) -> None:
        pass   # meaningless for a button

    def _value_str(self) -> str:
        return "pressed" if self._pressed else "—"


# ===========================================================================
# Demo  (__main__)
# ===========================================================================

if __name__ == "__main__":
    import numpy as np

    app = QApplication(sys.argv)

    class DemoWindow(QMainWindow):
        # --- typed declarations so Pylance sees them ---
        freq_ctrl:    SliderControl
        amp_ctrl:     SliderControl
        phase_ctrl:   RangeControl
        wave_ctrl:    ComboControl
        flags_ctrl:   ChecksetControl
        reset_ctrl:   ButtonControl

        def __init__(self):
            super().__init__()
            self.setWindowTitle("ParamControl demo")
            self.resize(960, 580)

            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.setHandleWidth(6)
            splitter.setStyleSheet("""
                QSplitter::handle { background-color: #404040; }
                QSplitter::handle:hover { background-color: #606060; }
            """)

            # ── control panel ──────────────────────────────────────────────
            ctrl_widget = QWidget()
            # ctrl_widget.setStyleSheet("background-color: #2b2b2b;")
            grid = QGridLayout(ctrl_widget)
            grid.setContentsMargins(8, 8, 8, 8)
            grid.setSpacing(6)

            panel = ControlPanel(grid, callback=self.on_change)

            # row 0 — full-width phase range
            self.phase_ctrl = RangeControl(
                "Phase window", min_val=0, max_val=360,
                default=(30, 270), unit="°",
                conv=lambda x: float(x),
            )
            panel.add(self.phase_ctrl, row=0, col=0, col_span=3)

            # row 1 — three sliders
            self.freq_ctrl = SliderControl(
                "Frequency", min_val=1, max_val=20, default=3,
                unit="Hz", conv=lambda x: float(x),
            )
            self.amp_ctrl = SliderControl(
                "Amplitude", min_val=1, max_val=50, default=10,
                conv=lambda x: x / 10.0,
            )
            panel.add(self.freq_ctrl, row=1, col=0)
            panel.add(self.amp_ctrl,  row=1, col=1)

            # combo in row 1 col 2
            self.wave_ctrl = ComboControl(
                "Waveform",
                options=["sine", "square", "sawtooth", "triangle"],
                default=0,
            )
            panel.add(self.wave_ctrl, row=1, col=2)

            # row 2 — checkset + button
            self.flags_ctrl = ChecksetControl(
                "Modifiers",
                labels=["invert", "noise", "DC offset"],
                defaults=[False, False, False],
            )
            self.reset_ctrl = ButtonControl("Actions", button_label="Reset all")
            self.reset_ctrl._btn.clicked.connect(self._reset_all)

            panel.add(self.flags_ctrl, row=2, col=0, col_span=2)
            panel.add(self.reset_ctrl, row=2, col=2)

            ctrl_widget.setMaximumHeight(240)

            # ── plot ───────────────────────────────────────────────────────
            plot_widget = QWidget()
            plot_widget.setStyleSheet("background-color: #1e1e1e;")
            plot_layout = QVBoxLayout(plot_widget)

            self._pw = pg.PlotWidget()
            self._pw.setBackground("#1e1e1e")
            self._pw.showGrid(x=True, y=True, alpha=0.3)
            self._curve = self._pw.plot(pen=pg.mkPen("#3ddc84", width=2))
            plot_layout.addWidget(self._pw)

            splitter.addWidget(ctrl_widget)
            splitter.addWidget(plot_widget)
            splitter.setSizes([220, 360])

            self.setCentralWidget(splitter)
            self.on_change()   # initial draw

        def on_change(self) -> None:
            freq   = self.freq_ctrl.value()
            amp    = self.amp_ctrl.value() / 10.0
            lo, hi = self.phase_ctrl.value()
            wave   = self.wave_ctrl.value()
            flags  = self.flags_ctrl.value()   # [invert, noise, dc_offset]

            x = np.linspace(0, 4 * np.pi, 800)

            # phase window
            ph_lo = np.deg2rad(lo)
            ph_hi = np.deg2rad(hi)
            x_plot = np.linspace(ph_lo, ph_hi, 800)

            if wave == "sine":
                y = amp * np.sin(freq * x_plot)
            elif wave == "square":
                y = amp * np.sign(np.sin(freq * x_plot))
            elif wave == "sawtooth":
                y = amp * (2 * (freq * x_plot / (2 * np.pi) % 1) - 1)
            else:  # triangle
                y = amp * (2 * np.abs(2 * (freq * x_plot / (2 * np.pi) % 1) - 1) - 1)

            if flags[0]:   # invert
                y = -y
            if flags[1]:   # noise
                y = y + np.random.normal(0, amp * 0.1, y.shape)
            if flags[2]:   # DC offset
                y = y + amp * 0.5

            self._curve.setData(x_plot, y)

            # show Δ on the amp slider as a demo
            self.amp_ctrl.set_delta(float(amp))

        def _reset_all(self) -> None:
            self.freq_ctrl.set_value(3)
            self.amp_ctrl.set_value(10)
            self.phase_ctrl.set_value((30, 270))
            self.wave_ctrl.set_value("sine")
            self.flags_ctrl.set_value([False, False, False])
            self.on_change()

    window = DemoWindow()
    window.show()
    sys.exit(app.exec())