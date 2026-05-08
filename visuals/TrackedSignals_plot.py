from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QApplication, QPushButton, QSplitter
from PySide6.QtCore import Qt

import numpy as np
import scipy
import pyqtgraph as pg
import matplotlib.pyplot as plt

import os, sys,types
from typing import Callable, Any
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
        self.setWindowTitle("Tracked signals")
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

        self.loadData_btn_ctrl = pctrl.ButtonControl("Load tracking data", button_label="Load tracking data")
        self.loadData_btn_ctrl._btn.clicked.connect(self.loadTrackingData)
        ctrl_panel.add(self.loadData_btn_ctrl,row=1,col=0)

        self.export_btn_ctrl = pctrl.ButtonControl("Export", button_label="Export CSV")
        self.export_btn_ctrl._btn.clicked.connect(self.exportData)
        ctrl_panel.add(self.export_btn_ctrl,row=1,col=1)
        
        self.plot_support_ctrl = pctrl.ChecksetControl("Plot support",labels=["Show points", "Show supports"], defaults=[False, False])
        ctrl_panel.add(self.plot_support_ctrl,row=1,col=2)

        self.sig_selection_ctrl = pctrl.DictTableControl(
                "Show signals",
                params={
                    "Show": ["targets","treshold","none"],
                    "Method": ["cMax 2D CFAR", "CA 2D CFAR"],
                    "A": True,
                    "B": False,
                    "C": True
                },
                orientation="horizontal",
            )
        ctrl_panel.add(self.sig_selection_ctrl, row=2, col=0,col_span=3)
        
        self.track_selection_ctrl = pctrl.DictTableControl(
                "Tracking params",
                params={
                    "Show": ["All","None","Select"],
                    "match W_range": 1.0,
                    "match W_azi": 2.0,
                    "gating dist": 3.0,
                    "show history": 2,
                    "Clustering": ["4 way NN ampl. mean"],
                    "Cost assignment": ["weighted norm 2"],
                    "Assignment": ["Gated Glob.NN"],
                },
                orientation="vertical",
            )
        ctrl_panel.add(self.track_selection_ctrl, row=0, col=4,row_span=3)



        ctrl_widget.setMaximumHeight(240)
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


        # tracking data retrieving function
        self.dataRetrievingFunc = None
        self.loadData_btn_ctrl.set_error("No data retrieve function assigned!")



        self.initDone = True

        # for multiple plots
        self.cmap = plt.get_cmap('tab10')
        

    def exportData(self):
        if not hasattr(self, "data2export"):
            return
        np.savetxt(
            "trackingOutput.csv",
            self.data2export,
            delimiter=",",
            header=self.dataExport_col_names,
            comments="",
            fmt="%.6f",
        )

    def assignDataRetrievingFunction(self,func):
        if callable(func):
            self.loadData_btn_ctrl.clear_state()
            self.dataRetrievingFunc = func
        else:
            self.loadData_btn_ctrl.set_error("STILL No data retrieve function assigned!")

    def loadTrackingData(self):
        if not callable(self.dataRetrievingFunc):
            self.loadData_btn_ctrl.set_error("Somehow the data retrieve function is not callable!")    
            return
        
        self.tracking_data = self.dataRetrievingFunc()

        if not isinstance(self.tracking_data, dict):
            self.loadData_btn_ctrl.set_warning(f"Data retrieve function retrieved {type(self.tracking_data).__name__ }, dict was expected!") 
            self.tracking_data = {}
        elif not self.tracking_data: # = empty dict
            self.loadData_btn_ctrl.set_warning(f"The retrieved dict is empty!") 
            return
        else:
            self.loadData_btn_ctrl.clear_state()

        # setup the controls
        ## tracks 
        IDs = list(self.tracking_data.keys())
        
        tracks_dict : dict[str, Any] = {"Show": ["Select","All","None"]} 
        for key in IDs:
            if str(key) in self.track_selection_ctrl.value().keys():
                tracks_dict[str(key)] = self.track_selection_ctrl.value()[str(key)]
            else: 
                tracks_dict[str(key)] = False

        self.track_selection_ctrl.set_newParams(tracks_dict)
        
        ## signals in tracks - assume all tracks have the same sigs
        signals = list(self.tracking_data[0].keys()) 

        if "frames" in signals:
            signals.remove("frames")
        else:
            print("No frames in the tracking data!!")

        signals_dict = {}
        for key in signals:
            signals_dict[key] = False
        
        self.sig_selection_ctrl.set_newParams(signals_dict)
        
            
        
        


    def update_onSliderMove(self):
        if not self.initDone or not hasattr(self, "params"):
            return

        if not self.initDone or not hasattr(self, "tracking_data"):
            return

        frame_begin, frame_end = self.frames_ctrl.value()
        frame_begin -= self.params[ "i_Frames_begin"]
        frame_end -= self.params[ "i_Frames_begin"]

        
        show_points = self.plot_support_ctrl.value()[0]
        show_supports = self.plot_support_ctrl.value()[1]

        self.plot.clear()
        i =0
        for key_ID, value_ID in self.track_selection_ctrl.value().items():
            if key_ID == "Show":
                continue
            key_ID = int(key_ID) # tracking data is enumerated by ints
            # print(f"ID:{key_ID} - {value_ID}")
            if value_ID == False:
                continue
            
            frames = self.tracking_data[key_ID]["frames"]
            frames_mask = (frames >= frame_begin) & (frames <= frame_end)
            frames_visible = frames[frames_mask]
            # frames = self.sig_selection_ctrl.value()["frames"]
            for key_sig, value_sig in self.sig_selection_ctrl.value().items():
                if key_sig == "frames":
                    continue
                if value_sig == False:
                    continue
                signal = self.tracking_data[key_ID][key_sig]
                signal_visible = signal[frames_mask]
                if len(signal_visible) == 0:
                    continue
                
                symbol = None
                if show_points:
                    symbol = "x"
                
                rcolor, g, b, acolor = self.cmap(i % 10)
                color = (int(rcolor*255), int(g*255), int(b*255), int(acolor*255))
                # color = pg.intColor(i, hues=12, values=200, sat=255)
                pen = pg.mkPen(color=color) 
                name = f"ID:{key_ID} {key_sig}"
                self.plot.plot(frames_visible, signal_visible, pen=pen,name=name,symbol=symbol,symbolPen=pen,symbolBrush=pen.color(),symbolSize = 6)
                i+=1

                if show_supports:
                    support_y = np.max(signal_visible) * 1.05
                    disc_idx = np.where(np.diff(frames_visible) > 1)[0] # list of discontinuities
                    frame_segments = np.split(frames_visible, disc_idx)

                    for segment_frames in frame_segments:
                        if len(segment_frames) == 0:
                            continue

                        self.plot.plot(segment_frames,np.full(len(segment_frames), support_y),pen=pen) # draw the line
                        self.plot.plot([segment_frames[0], segment_frames[-1]],[support_y, support_y],pen=None,symbol="|",symbolPen=pen,symbolBrush=pen.color(),symbolSize=12) # draw the end caps

        # detr_freq_idx = self.detr_ctrl.value()
        
        # if detr_freq_idx == 0:
        #     detr_freq = 0
        # else:
        #     detr_freq = np.pow(2,detr_freq_idx-1)/100

        return
        # # new_data = #self.penteract[frame_begin:frame_end, dopp_begin:dopp_end, range_begin:range_end, val_ele, azi_begin:azi_end]
        # if new_data.size == 0:
        #     return
        # (N_Frames,N_Doppler,N_Range,N_Azi)
        # new_data = aggregate(new_data,axis=1,ag_type=dopp_aggregation,keepDims=False)
        # # (N_Frames,N_Range,N_Azi)
        # new_data = aggregate(new_data,axis=1,ag_type=range_aggregation,keepDims=True)
        # new_data = aggregate(new_data,axis=2,ag_type=azi_aggregation,keepDims=True)
        # (N_Frames,N_Range,N_Azi)

        # new_data = self._detrending(detr_freq/10,new_data)

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
                # if new_data.shape[1] < (range_end-range_begin):
                #     name += f"r{range_begin}:{range_end}_"
                # else:
                #     name += f"r{range_begin+r}_"

                # if new_data.shape[2] < (azi_end-azi_begin):
                #     name += f"a{azi_begin}:{azi_end}"
                # else:
                #     name += f"a{azi_begin+a}"

                # # print(name)
                # self.plot.plot(t, new_data[:,r,a], pen=pen,name=name)
                # if(i==0):
                #     self.data2export = np.column_stack((t, new_data[:,r,a]))
                #     self.dataExport_col_names = [str]*(new_data.shape[1]*new_data.shape[2] +1)
                #     self.dataExport_col_names[0] = "t"
                #     self.dataExport_col_names[1] = name
                # else:
                #     self.data2export = np.column_stack((t, new_data[:,r,a]))
                #     self.dataExport_col_names[i+1] = name

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

    def update_newParams(self, params):
        self.initDone = False
        
        self.params = params

        self.frames_ctrl.set_range(params["i_Frames_begin"], params["i_Frames_end"] - 1)
        self.frames_ctrl.set_conv(lambda x: x * params["frame_index2time"])
        self.frames_ctrl.set_delta(params["frame_index2time"])


        self.initDone = True
        self.update_onSliderMove()


if __name__ == "__main__":
    app = QApplication([])
    window = PlotWindow()
    window.resize(800, 600)
    window.show()
    app.exec()
