from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QApplication, QPushButton, QSplitter
from PySide6.QtCore import Qt

import numpy as np
import scipy
import pyqtgraph as pg
import matplotlib.pyplot as plt

import os, sys,types
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import visuals.param_controls as pctrl


def phaseUnwrapping(data):
    phase = np.angle(data)
    return np.unwrap(phase, axis=0)

def aggregate(data :np.ndarray, axis :int, ag_type:str,keepDims :bool = False) -> np.ndarray :
    ops = {
        "Single bin (min)": None,
        "mean": np.mean,
        "max": np.max,
        "Show all independent": None
    }

    if ag_type == "Single bin (min)":
        slicer = [slice(None)] * data.ndim
        slicer[axis] = 0
        result = data[tuple(slicer)]
        
        if keepDims:
            result = np.expand_dims(result, axis=axis)
        return result

    if ag_type == "Show all independent":
        return data

    func = ops.get(ag_type.lower())
    if func is None:
        raise ValueError(f"Unknown aggregation type: {ag_type}")

    return func(data, axis=axis, keepdims=keepDims)

class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initDone = False
        self.setWindowTitle("Phase Unwrapping")
        self.resize(960, 620)

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

        self.frames_ctrl = pctrl.RangeControl(
            "Frame range", min_val=0, max_val=100,
            default=(10, 50), unit="s",
        )
        ctrl_panel.add(self.frames_ctrl, row=0, col=0, col_span=2)

        self.doppler_aggregation_ctrl = pctrl.ComboControl(
                "Doppler aggregation selector",
                options=[ "Single bin (min)" , "Mean", "Max"],
                default=0,
            )
        ctrl_panel.add(self.doppler_aggregation_ctrl, row=0, col=2)

        self.doppler_ctrl = pctrl.RangeControl(
            "Doppler range", min_val=0, max_val=100,
            default=(0, 50), unit="Hz",
        )
        ctrl_panel.add(self.doppler_ctrl, row=0, col=3)


        self.range_ctrl = pctrl.RangeControl(
            "Range selector", min_val=0, max_val=100,
            default=(15,18), unit="m",
        )
        ctrl_panel.add(self.range_ctrl, row=1, col=0,col_span=2)

        self.azi_ctrl = pctrl.RangeControl(
            "Azimuth selector", min_val=0, max_val=8,
            default=(4,6), unit="°",
        )
        ctrl_panel.add(self.azi_ctrl, row=1, col=2)

        self.ele_ctrl = pctrl.SliderControl(
            "Elevation selector", min_val=0, max_val=100,
            default=0, unit="°",
        )
        ctrl_panel.add(self.ele_ctrl, row=1, col=3)

        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.exportData)
        grid.addWidget(self.export_button, 2, 3)

        self.range_aggregation_ctrl = pctrl.ComboControl(
            "Range aggregation selector",
            options=[ "Show all independent" , "Mean", "Max"],
            default=0,
        )
        ctrl_panel.add(self.range_aggregation_ctrl, row=2, col=0)

        self.azi_aggregation_ctrl = pctrl.ComboControl(
            "Azimuth aggregation selector",
            options=[ "Show all independent" , "Mean", "Max"],
            default=0,
        )
        ctrl_panel.add(self.azi_aggregation_ctrl, row=2, col=2)

        
        self.detr_ctrl = pctrl.SliderControl(
            "Detrending freq selector", min_val=0, max_val=20,
            default=0, unit="Hz",
        )
        ctrl_panel.add(self.detr_ctrl, row=2, col=1)


        ctrl_widget.setMaximumHeight(160)
        splitter.addWidget(ctrl_widget)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.legend = self.plot.addLegend() 
        plot_layout.addWidget(self.plot)

        splitter.addWidget(plot_widget)
        splitter.setSizes([180, 420])

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)

        self.initDone = True

        # for multiple plots
        self.cmap = plt.get_cmap('tab10')
        

    def exportData(self):
        if not hasattr(self, "data2export"):
            return
        np.savetxt(
            "phaseUnwrapingOutput.csv",
            self.data2export,
            delimiter=",",
            header=self.dataExport_col_names,
            comments="",
            fmt="%.6f",
        )

    def update_onSliderMove(self):
        if not self.initDone or not hasattr(self, "params"):
            return

        frame_begin, frame_end = self.frames_ctrl.value()
        frame_begin -= self.params[ "i_Frames_begin"]
        frame_end -= self.params[ "i_Frames_begin"]

        range_begin, range_end = self.range_ctrl.value()
        range_aggregation = self.range_aggregation_ctrl.value()
        
        azi_begin, azi_end = self.azi_ctrl.value()
        azi_aggregation = self.azi_aggregation_ctrl.value()
        
        val_ele = self.ele_ctrl.value()

        detr_freq_idx = self.detr_ctrl.value()
        
        if detr_freq_idx == 0:
            detr_freq = 0
        else:
            detr_freq = np.pow(2,detr_freq_idx-1)/100

        dopp_begin, dopp_end =  self.doppler_ctrl.value()
        dopp_aggregation = self.doppler_aggregation_ctrl.value()


        new_data = self.penteract[frame_begin:frame_end, dopp_begin:dopp_end, range_begin:range_end, val_ele, azi_begin:azi_end]
        if new_data.size == 0:
            return
        # (N_Frames,N_Doppler,N_Range,N_Azi)
        new_data = aggregate(new_data,axis=1,ag_type=dopp_aggregation,keepDims=False)
        # (N_Frames,N_Range,N_Azi)
        new_data = aggregate(new_data,axis=1,ag_type=range_aggregation,keepDims=True)
        new_data = aggregate(new_data,axis=2,ag_type=azi_aggregation,keepDims=True)
        # (N_Frames,N_Range,N_Azi)

        new_data = self._detrending(detr_freq/10,new_data)

        time_begin = frame_begin * self.params["frame_index2time"]
        time_end = frame_end * self.params["frame_index2time"]
        t = np.linspace(time_begin, time_end, frame_end - frame_begin)

        self.plot.clear()

        i = 0
        for r in range(new_data.shape[1]):
            for a in range(new_data.shape[2]):
                rcolor, g, b, acolor = self.cmap(i % 10)
                color = (int(rcolor*255), int(g*255), int(b*255), int(acolor*255))
                # color = pg.intColor(i, hues=12, values=200, sat=255)
                pen = pg.mkPen(color=color) 
                name = ""
                if new_data.shape[1] < (range_end-range_begin):
                    name += f"r{range_begin}:{range_end}_"
                else:
                    name += f"r{range_begin+r}_"

                if new_data.shape[2] < (azi_end-azi_begin):
                    name += f"a{azi_begin}:{azi_end}"
                else:
                    name += f"a{azi_begin+a}"

                # print(name)
                self.plot.plot(t, new_data[:,r,a], pen=pen,name=name)
                if(i==0):
                    self.data2export = np.column_stack((t, new_data[:,r,a]))
                    self.dataExport_col_names = [str]*(new_data.shape[1]*new_data.shape[2] +1)
                    self.dataExport_col_names[0] = "t"
                    self.dataExport_col_names[1] = name
                else:
                    self.data2export = np.column_stack((t, new_data[:,r,a]))
                    self.dataExport_col_names[i+1] = name

                i += 1
        # new_data_2d = new_data.reshape(new_data.shape[0], -1)
        # self.data2export = np.column_stack((t, new_data_2d))

        # r = new_data.shape[1]
        # azi = new_data.shape[2]
        # self.dataExport_col_names = [f"r{bi}_azi{ci}" for bi in range(r) for ci in range(azi)]

    def _azi_to_deg(self, idx: int) -> float:
        span = self.params["DoA_azi_range_degs"]
        count = self.params["DoA_azi_N_elements"]
        step = (2 * span) / count
        offset = -span + step / 2
        return idx * step + offset

    def _ele_to_deg(self, idx: int) -> float:
        span = self.params["DoA_ele_range_degs"]
        count = self.params["DoA_ele_N_elements"]
        step = (2 * span) / count
        offset = -span + step / 2
        return idx * step + offset

    def _detrending(self, freq : float, phase_unwrp : np.ndarray):
        
        if(freq == 0):
            return phase_unwrp

        fs = 1/ self.params["frame_index2time"]
        # Cutoff frequency
        fc = freq  # Hz
        w = fc / (fs / 2) # Normalize cutoff (Nyquist frequency = fs/2)

        # Design Butterworth high-pass filter
        order = 4  # good default
        b, a = scipy.signal.butter(order, w, btype='highpass')

        # filtered_signal = scipy.signal.filtfilt(b, a, phase_unwrp,axis=0)
        filtered_signal = scipy.signal.filtfilt(b, a, phase_unwrp,axis=0)
        return filtered_signal

        pass

    def update_newData(self, data, params):
        self.initDone = False
        self.penteract = phaseUnwrapping(data)
        self.params = params

        self.frames_ctrl.set_range(params["i_Frames_begin"], params["i_Frames_end"] - 1)
        self.frames_ctrl.set_conv(lambda x: x * params["frame_index2time"])
        self.frames_ctrl.set_delta(params["frame_index2time"])

        self.doppler_ctrl.set_range(params["i_Doppler_begin"],params["i_Doppler_end"]-1 )
        self.doppler_ctrl.set_conv(lambda x: float(params["doppler_index2freq"]*x) )

        self.range_ctrl.set_range(params["i_Range_begin"], params["i_Range_end"] - 1)
        self.range_ctrl.set_conv(lambda x: x * params["range_index2dist"])

        self.azi_ctrl.set_range(0, params["DoA_azi_N_elements"] - 1)
        self.azi_ctrl.set_conv(self._azi_to_deg)

        self.ele_ctrl.set_range(0, params["DoA_ele_N_elements"] - 1)
        self.ele_ctrl.set_conv(self._ele_to_deg)

        self.detr_ctrl.set_range(0,20) 
        self.detr_ctrl.set_conv(lambda x:                        
                                    np.pow(2,x-1)/100 if(x>0) 
                                    else 0
                                ) 
        self.detr_ctrl._value_str = types.MethodType( # printing needs special attention
            lambda self: f"idx {self.value()} ~ {self._conv(self.value()):.3f}"
            if self._conv else f"idx {self.value()}",
            self.detr_ctrl
        )

        if params["Doppler_processing"] == "None":
            self.doppler_ctrl.set_warning("Data is not doppler processed!")
            self.doppler_aggregation_ctrl.set_warning("Data is not doppler processed!")
        else:
            self.doppler_ctrl.clear_state()
            self.doppler_aggregation_ctrl.clear_state()

        self.initDone = True
        self.update_onSliderMove()


if __name__ == "__main__":
    app = QApplication([])
    window = PlotWindow()
    window.resize(800, 600)
    window.show()
    app.exec()
