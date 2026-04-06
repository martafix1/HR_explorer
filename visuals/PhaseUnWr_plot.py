from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QApplication, QPushButton
)
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

import sys

import numpy as np
import pyqtgraph as pg
from superqt import QRangeSlider
import visuals.utils as utils

"""
DOCS:

penteract: 4D ndarray in form data[N_Frames,N_range,N_Azi,N_ele] 

sliderLink: {"i_range":0,  "i_Frames_begin": 0, "i_Frames_end": 69 }

highlightLink: {"i_azi":0, "i_ele":0}


Params: dict {  "range_index2dist"  : 6e-2,
                "frame_index2time"  : 5e-2,
                "azi_points" : np.ndarray...,
                "ele_points" : np.ndarray...,
                "i_Range_begin": 0, "i_Range_end": 67,
                "i_Doppler_begin":0,"i_Doppler_end":32,
                "i_Frames_begin": 0, "i_Frames_end": 69,
                
                "DoA_azi_N_elements":24,
                "DoA_ele_N_elements":12,
                "DoA_azi_range_degs":90, "DoA_ele_range_degs":30,
                } 
"""

# --- reusable slider block ---
def make_slider_block(name, slider, value_label):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(5, 2, 5, 2)

    # top row
    top = QHBoxLayout()
    label = QLabel(name)

    top.addWidget(label)
    top.addStretch()
    top.addWidget(value_label)

    layout.addLayout(top)

    # bottom row (slider)
    layout.addWidget(slider)

    container.setMaximumHeight(60)
    return container

def phaseUnwrapping(data4D):
    phase = np.angle(data4D)
    data4D = np.unwrap(phase, axis=0)

    return data4D



class PlotWindow(QWidget):
    def __init__(self):
        self.initDone = False
        super().__init__()

        self.setWindowTitle("Plot Window")
        self.resize(700, 500)

        main_layout = QVBoxLayout(self)

        # =========================
        # --- TOP CONTROL PANEL ---
        # =========================
        controls = QVBoxLayout()
        self._scrollWheel_filter = utils.SliderWheelFilter()


        # --- Frame range (full width) ---
        self.frame_range = QRangeSlider(Qt.Horizontal) # type: ignore
        self.frame_range.setRange(0, 100)
        # self.frame_range.installEventFilter(self._scrollWheel_filter)
        self.frame_range_label = QLabel()
        self.frame_range.valueChanged.connect(
            # lambda v: self.frame_range_label.setText(f"{v[0]} - {v[1]}")
            self.update_onSliderMove
        )
        self.frame_range.setValue((10, 50))
        self.frame_range_label.setText("10 - 50")
        

        self.frame_range.setStyleSheet("""
        QRangeSlider {
            qproperty-barColor: #3ddc84;
        }
        """)

        # self.frame_range.setBarVisible(True)
       
        controls.addWidget(
            make_slider_block(
                "Frame start - Frame end",
                self.frame_range,
                self.frame_range_label
            )
        )

        # --- grid for remaining sliders (2 rows x 3 cols) ---
        grid = QGridLayout()

        def make_named_slider(attr_name, label_text, row, col):
            slider = QSlider(Qt.Horizontal) # type: ignore
            slider.setRange(0, 100)
            slider.installEventFilter(self._scrollWheel_filter)

            value_label = QLabel()

            slider.valueChanged.connect(
                #lambda v, lbl=value_label: lbl.setText(str(v))
                self.update_onSliderMove
            )

            slider.setValue(50)
            value_label.setText("50")

            setattr(self, attr_name, slider)
            setattr(self, f"{attr_name}_label", value_label)

            block = make_slider_block(label_text, slider, value_label)
            grid.addWidget(block, row, col)

        # row 1
        self.azi_slider : QSlider
        self.ele_slider : QSlider
        self.range_slider : QSlider

        self.azi_slider_label : QLabel
        self.ele_slider_label : QLabel
        self.range_slider_label : QLabel

        make_named_slider("azi_slider", "Azimuth", 0, 0)
        make_named_slider("ele_slider", "Elevation", 0, 1)
        make_named_slider("range_slider", "Range", 0, 2)

        self.export_button : QPushButton
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.exportData)

        grid.addWidget(self.export_button,0,4)

        controls.addLayout(grid)

        # limit total height of control area
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumHeight(180)

        main_layout.addWidget(controls_widget)


        # --------- PLOT ----------
        self.plot = pg.PlotWidget()
        main_layout.addWidget(self.plot)

        self.plot.setBackground("w")
        # grid on
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # placeholder data
        x = np.linspace(0, 10, 500)
        y = np.sin(x)

        self.plot.plot(x, y, pen="b")

        self.initDone = True

    def exportData(self):
        
        np.savetxt(
            "phaseUnwrapingOutput.csv",
            self.data2export,
            delimiter=",",
            header="t [s], phi [rad]",
            comments="",   # prevents '#' before header
            fmt="%.3f"       # or "%d", "%.3f", etc.
        )


    def update_onSliderMove(self):
        if self.initDone == False:
            return
        val_range = self.range_slider.value()
        range_m = val_range*self.params["range_index2dist"]
        self.range_slider_label.setText(f"idx: {val_range} ~ {range_m :.2f} m")

        val_frame_begin,val_frame_end = self.frame_range.value()
        frameTimes_s = (val_frame_begin* self.params["frame_index2time"],val_frame_end* self.params["frame_index2time"]) 
        self.frame_range_label.setText(f"idx: {val_frame_begin} - {val_frame_end} ~ {frameTimes_s[0] :.2f} - {frameTimes_s[1] :.2f} s")
        # sliders = {"Frame": 10,"Range":17}

        val_azi = self.azi_slider.value()
        azi_index2deg_scale =  ((2*self.params["DoA_azi_range_degs"]) / self.params["DoA_azi_N_elements"])
        azi_index2deg_offset =  -1*self.params["DoA_azi_range_degs"] + azi_index2deg_scale/2
        azi_deg = val_azi*azi_index2deg_scale + azi_index2deg_offset
        self.azi_slider_label.setText(f"idx: {val_azi} - {azi_deg} deg")
        
        
        val_ele = self.ele_slider.value()
        ele_index2deg_scale =  ((2*self.params["DoA_ele_range_degs"]) / self.params["DoA_ele_N_elements"])
        ele_index2deg_offset =  -1*self.params["DoA_ele_range_degs"] + ele_index2deg_scale/2
        ele_deg = val_ele*ele_index2deg_scale + ele_index2deg_offset
        self.ele_slider_label.setText(f"idx: {val_ele} - {ele_deg} deg")

        new_data = self.data4D[val_frame_begin:val_frame_end, val_range,val_ele,val_azi]
        
        t = np.linspace(frameTimes_s[0],frameTimes_s[1], val_frame_end-val_frame_begin)
        self.plot.clear()
        self.plot.plot(t, new_data, pen="b")

        self.data2export = np.column_stack((t, new_data))

            

    def update_newData(self,data,params):
        self.initDone = False
        self.data4D = phaseUnwrapping(data)
        self.params = params
        

        self.frame_range.setRange(params[ "i_Frames_begin"], params["i_Frames_end"]-1)
        self.azi_slider.setRange(0, params["DoA_azi_N_elements"]-1)
        self.ele_slider.setRange(0, params["DoA_ele_N_elements"]-1)
        self.range_slider.setRange(0, params["i_Range_end"]-1)


        self.initDone = True
        self.update_onSliderMove()
        pass



if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = PlotWindow()
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())