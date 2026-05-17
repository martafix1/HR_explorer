from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QApplication, QPushButton, QSplitter
from PySide6.QtCore import Qt

import numpy as np
import scipy
import pyqtgraph as pg
import matplotlib.pyplot as plt

import re # regex
import os, sys,types
from typing import Callable, Any
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import visuals.param_controls as pctrl
import visuals.SpectraInspector as SpectraInspector
import processing.slidingFFT as slidingFFT

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
        self.setWindowTitle("Vital extraction from tracked signals")
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

        self.second_plot_ctrl = pctrl.ChecksetControl("Second plot",labels=["Show second plot", "Link X axes", "Link Y axes"], defaults=[False, True, False])
        ctrl_panel.add(self.second_plot_ctrl,row=0,col=2)

        self.export_btn_ctrl = pctrl.ButtonControl("Export", button_label="Export CSV")
        self.export_btn_ctrl._btn.clicked.connect(self.exportData)
        ctrl_panel.add(self.export_btn_ctrl,row=1,col=1)

        self.spectra_btn_ctrl = pctrl.ButtonControl("Spectra", button_label="Inspect spectra")
        self.spectra_btn_ctrl._btn.clicked.connect(self.openSpectraInspector)
        ctrl_panel.add(self.spectra_btn_ctrl,row=4,col=3)
        
        self.plot_support_ctrl = pctrl.ChecksetControl("Plot support",labels=["Show points", "Show supports"], defaults=[False, False])
        ctrl_panel.add(self.plot_support_ctrl,row=1,col=2)


        paramsDict = self.buildSignalControlDict()
        self.sig_processing1_ctrl = pctrl.DictTableControl_composite(
                "Processing",
                params=paramsDict,
                headers=["PLOT 1","PLOT 2", "Process","Input", "Param1","Param2","Param3","Param4"],
                orientation="horizontal",
            )
        ctrl_panel.add(self.sig_processing1_ctrl, row=2, col=0,col_span=3,row_span=4)

        selected_methods = [] # manually save the last methods 
        for i,name in enumerate(self.sig_processing1_ctrl.value().keys()):
            selected_methods.append(self.sig_processing1_ctrl.value()[name]["Method"]) 
        self.sig_processing1_ctrl.previous_methods = selected_methods    


        
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


        paramsDict = self.buildRefSigControlDict()
        self.ref_signal_ctrl = pctrl.DictTableControl_composite(
                "Reference signal plotting",
                params=paramsDict,
                headers=["PLOT 1","PLOT 2" ],
                orientation="vertical",
            )
        ctrl_panel.add(self.ref_signal_ctrl, row=0, col=3,col_span=1,row_span=3)
    


        # ctrl_widget.setMaximumHeight(4*80)
        ctrl_widget.setMaximumHeight(5*80)
        splitter.addWidget(ctrl_widget)

        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        plot_splitter.setHandleWidth(6)
        plot_splitter.setStyleSheet("""
            QSplitter::handle { background-color: #404040; }
            QSplitter::handle:hover { background-color: #606060; }
        """)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.legend = self.plot.addLegend()
        plot_splitter.addWidget(self.plot)

        self.plot2 = pg.PlotWidget()
        self.plot2.setBackground("w")
        self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.legend2 = self.plot2.addLegend()
        self.plot2.setVisible(False)
        plot_splitter.addWidget(self.plot2)

        splitter.addWidget(plot_splitter)
        splitter.setSizes([180, 420])

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)


        # tracking data retrieving function
        self.dataRetrievingFunc = None
        self.loadData_btn_ctrl.set_error("No data retrieve function assigned!")



        self.initDone = True

        # for multiple plots
        self.cmap = plt.get_cmap('tab10')
        self._update_second_plot_controls_state()

        self.spectraInspector = None

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
        first_key = next(iter(self.tracking_data))
        signals = list(self.tracking_data[first_key].keys())


        if "frames" in signals:
            signals.remove("frames")
        else:
            print("No frames in the tracking data!!")

        # signals_dict = {}
        # for key in signals:
        #     signals_dict[key] = False
        
        # self.sig_selection_ctrl.set_newParams(signals_dict)

        # current_track2 = self.track_selection2_ctrl.value()
        # tracks2_dict : dict[str, Any] = {"Show": ["Select","All","None"]}
        # for key in IDs:
        #     if str(key) in current_track2.keys():
        #         tracks2_dict[str(key)] = current_track2[str(key)]
        #     else:
        #         tracks2_dict[str(key)] = False

        # current_sig2 = self.sig_selection2_ctrl.value()
        # signals2_dict = {}
        # for key in signals:
        #     signals2_dict[key] = current_sig2.get(key, False)

        # self.track_selection2_ctrl.set_newParams(tracks2_dict)
        # self.sig_selection2_ctrl.set_newParams(signals2_dict)

        self._update_second_plot_controls_state()

    def _enabled_track_ids_for_inspector(self) -> list[str]:
        if not hasattr(self, "tracking_data"):
            return []

        selected = self.track_selection_ctrl.value()
        show_mode = selected.get("Show", "Select")
        if show_mode == "All":
            return sorted(str(key) for key in self.tracking_data.keys())

        ids: list[str] = []
        for key, value in selected.items():
            if key == "Show":
                continue
            if value:
                ids.append(str(key))
        return ids or sorted(str(key) for key in self.tracking_data.keys())

    def openSpectraInspector(self):
        if not hasattr(self, "tracking_data") or not self.tracking_data:
            self.spectra_btn_ctrl.set_warning("Load tracking data first")
            return
        if not hasattr(self, "params"):
            self.spectra_btn_ctrl.set_warning("No params available")
            return
        
        self.spectra_btn_ctrl.clear_state()

        if self.spectraInspector is None:
            self.spectraInspector = SpectraInspector.SpectraInspectorWindow()

        self.spectraInspector.update_context(
            tracking_data=self.tracking_data,
            params=self.params,
            HRs=getattr(self, "HRs", {}),
            vital_processing_values=self.sig_processing1_ctrl.value(),
            vital_signal_processor=self.signalProcessing,
            enabled_track_ids=self._enabled_track_ids_for_inspector(),
        )
        self.spectraInspector.show()
        self.spectraInspector.raise_()
        self.spectraInspector.activateWindow()

    def signalProcessing(self,signal: np.ndarray | None = None, 
                          method: str | None = None,
                          m_params: dict  = {}):
        capabalities = {
                "show original": {},    
                "detrending": {"cutoff [Hz]": 0.1, "order": 4 },
                "show trend": {"cutoff [Hz]": 0.1, "order": 4 },
                "low pass": {"cutoff [Hz]": 0.1, "order": 4 },
                "band pass": {"cutoff0 [Hz]": 0.2, "cutoff1 [Hz]":0.6, "order": 4},
                "inst freq Hilbert": {},
                "inst ampl Hilbert": {},
                "inst f slidingFFT": {  "initFrames": 20, "stepFrames":10, "sig_sample":80,
                                        "fixedFFT_size":200, "freqRangeStart":0.1, "freqRangeStop":2.0,
                                        "parabolicInterpolation": False, "window": ["rect", "hann", "hamming","blackman"]  },
                "inst A slidingFFT": {  "initFrames": 20, "stepFrames":10, "sig_sample":80,
                                        "fixedFFT_size":200, "freqRangeStart":0.1, "freqRangeStop":2.0,
                                        "parabolicInterpolation": False, "window": ["rect", "hann", "hamming","blackman"]  }
            }
        if method is None or signal is None:# describe avalible methods
            return capabalities

        if method in capabalities.keys():
            if method == "show original":
                return signal
            
            if method == "detrending":
                fs = 1/self.params["frame_index2time"]  # Hz sampling freq

                # Cutoff frequency
                fc = m_params["cutoff [Hz]"] # Hz
                w = fc / (fs / 2) # Normalize cutoff (Nyquist frequency = fs/2)

                # Design Butterworth high-pass filter
                order =  m_params["order"]  # good default
                b, a = scipy.signal.butter(order, w, btype='highpass')
                
                filtered_signal = scipy.signal.lfilter(b, a, signal)
                return filtered_signal
            
            if method == "show trend":
                fs = 1/self.params["frame_index2time"]  # Hz sampling freq

                # Cutoff frequency
                fc = m_params["cutoff [Hz]"] # Hz
                w = fc / (fs / 2) # Normalize cutoff (Nyquist frequency = fs/2)

                # Design Butterworth high-pass filter
                order =  m_params["order"]  # good default
                b, a = scipy.signal.butter(order, w, btype='highpass')
                
                filtered_signal = scipy.signal.lfilter(b, a, signal)
                return filtered_signal-signal # the difference is the key
            
            if method == "low pass":
                fs = 1/self.params["frame_index2time"]  # Hz sampling freq

                # Cutoff frequency
                fc = m_params["cutoff [Hz]"] # Hz
                w = fc / (fs / 2) # Normalize cutoff (Nyquist frequency = fs/2)

                # Design Butterworth high-pass filter
                order =  m_params["order"]  # good default
                b, a = scipy.signal.butter(order, w, btype='lowpass')
                
                filtered_signal = scipy.signal.lfilter(b, a, signal)
                return filtered_signal # the difference is the key
            
            if method == "band pass":
                fs = 1/self.params["frame_index2time"]  # Hz sampling freq
                fc = ( m_params["cutoff0 [Hz]"],m_params["cutoff1 [Hz]"])  # Hz, cutoff freq
                order = m_params["order"]

                b, a = scipy.signal.butter(order, fc, btype='bandpass',fs = fs)
                filtered_signal = scipy.signal.lfilter(b, a, signal)
                return filtered_signal
            
            if method == "inst freq Hilbert":
                fs = 1/self.params["frame_index2time"]  # Hz sampling freq
                analytic_signal = scipy.signal.hilbert(signal)
                phase_analytic_signal = np.unwrap(np.angle(analytic_signal))
                inst_freq = np.diff(phase_analytic_signal) * fs / (2 * np.pi)
                expanded = np.concatenate([inst_freq, inst_freq[-1:]]) # diff costs us a element, repeat the last so it can be ploted
                return expanded
            
            if method == "inst ampl Hilbert":
                fs = 1/self.params["frame_index2time"]  # Hz sampling freq
                analytic_signal = scipy.signal.hilbert(signal)
                inst_amplitude = np.abs(analytic_signal)
                return inst_amplitude
            
            if method == "inst A slidingFFT":
                fs = 1/self.params["frame_index2time"]
                f,a = slidingFFT.instantaneous_analysis_FFT(signal,fs,m_params["initFrames"],m_params["stepFrames"],m_params["sig_sample"],
                                                            m_params["fixedFFT_size"],m_params["freqRangeStart"],m_params["freqRangeStop"],
                                                            m_params["parabolicInterpolation"], m_params["window"])
                return a

            if method == "inst f slidingFFT":
                fs = 1/self.params["frame_index2time"]
                f,a = slidingFFT.instantaneous_analysis_FFT(signal,fs,m_params["initFrames"],m_params["stepFrames"],m_params["sig_sample"],
                                                            m_params["fixedFFT_size"],m_params["freqRangeStart"],m_params["freqRangeStop"],
                                                            m_params["parabolicInterpolation"], m_params["window"])
                return f

            else:
                raise Exception("Method not implemented") 
        else:
            raise Exception("Wrong signal processing method selected")



    def buildRefSigControlDict(self):
        # layout = ["Name","Plot 1", "Plot 2"]
        if not hasattr(self,"HRs"):
            return {}
        if self.HRs is None:
            return {}

        output = {}
        for refName in self.HRs.keys():
            inner_output = {}
            inner_output["Plot 1"] = False
            inner_output["Plot 2"] = False
            output[refName] = inner_output
        
        return output
    
    # def buildID_selectControlDict(self):
    #     layout = ["Name","Plot 1", "Plot 2"]

    #     output = {}
    #     for refName in self.HRs.keys():
    #         inner_output = {}
    #         inner_output["Plot 1"] = False
    #         inner_output["Plot 2"] = False
    #         output[refName] = inner_output
        
    #     return output



    def buildSignalControlDict(self, control_values: dict | None = None, lastMethods: list | None = None , theControl_forWarnRaising = None):
        capabalities_dict : dict[str, dict[str, Any]] =  self.signalProcessing(signal=None)  # type: ignore
        nLines = 5
        # layout = {"Plot": False, "Method": ["None"], "Input": ["hihi"]}
        


        if control_values is None:
            selected_methods = ["None","detrending","band pass","None","None"] # needs to be inputed later
        else:
            selected_methods = []
            for i,name in enumerate(control_values.keys()):
                # print(control_values[name]["Method"])
                selected_methods.append(control_values[name]["Method"]) 

        method_changed = []
        
        for i, method in enumerate(selected_methods):
            if lastMethods is not None:
                if method == lastMethods[i]:
                    method_changed.append(False)
                else:
                    method_changed.append(True)    
            else:
                method_changed.append(True)


        output = {}
        for i in range(nLines):
            line_name = f"Line {i+1}"
            previous_line = (control_values[line_name] if control_values is not None else {})


            inner_output = {}
            inner_output["Plot 1"] = previous_line.get("Plot 1", False)
            inner_output["Plot 2"] = previous_line.get("Plot 2", False)

            # Method
            methods = [selected_methods[i]] + ["None"] + list(capabalities_dict.keys())
            inner_output["Method"] = methods

            # Chaining
            selected_chain = previous_line.get("Input", "Ph. unwrap")
            match = re.fullmatch(r"Line (\d+)", selected_chain)
            if match:
                selectedLine_i = int(match.group(1))
                if selectedLine_i >= i+1: # cannot select line in future
                    selected_chain = "Ph. unwrap"
                    if theControl_forWarnRaising is not None:
                        theControl_forWarnRaising.set_warning("Cannot have a line with greater index as input")
                        theControl_forWarnRaising._warning_timer.start(1800)

            chain_list = [selected_chain] + ["Ph. unwrap"]
            for j in range(nLines):
                chain_list.append(f"Line {j+1}")
            inner_output["Input"] = chain_list

            # fill in the method params
            if selected_methods[i] in capabalities_dict:
                params = capabalities_dict[selected_methods[i]]
                if method_changed[i]:
                    for name, val in params.items():
                        inner_output[name] = val
                else:
                    for name, default_value in params.items():
                        inner_output[name] = previous_line.get(name, default_value)

            output[line_name] = inner_output

        return output


    def _update_second_plot_controls_state(self):
        show_second, lock_x, lock_y = self.second_plot_ctrl.value()

        if show_second:
            self.plot2.setVisible(True)
            self.legend2.setVisible(True)
        else:
            self.plot2.setVisible(False)
            self.legend2.setVisible(False)

        if lock_x and show_second:
            self.plot2.getPlotItem().setXLink(self.plot.getPlotItem()) # type: ignore
        else:
            self.plot2.getPlotItem().setXLink(None) # type: ignore

        if lock_y and show_second:
            self.plot2.getPlotItem().setYLink(self.plot.getPlotItem()) # type: ignore
        else:
            self.plot2.getPlotItem().setYLink(None) # type: ignore



    def _plotHRs(self,frame_begin,frame_end):

        refSigSelection = self.ref_signal_ctrl.value()

        iref = 0 
        for refName, val in refSigSelection.items():
    
            symbol = None #"x" if show_points else None
            rcolor, g, b, acolor = self.cmap(iref % 10)
            color = (int(rcolor * 255), int(g * 255), int(b * 255), int(acolor * 255))
            pen = pg.mkPen(color=color, style = Qt.DashDotLine)
            name = f"ID:{refName}"

            frames = self.HRs[refName]["frames"]
            signal = self.HRs[refName]["hr"]

            frames_mask = (frames >= frame_begin) & (frames <= frame_end)
            frames_visible = frames[frames_mask]
            signal_visible = signal[frames_mask]

            if(val["Plot 1"]):
                self.plot.plot(
                    frames_visible,
                    signal_visible,
                    pen=pen,
                    name=name,
                    symbol=symbol,
                    symbolPen=pen,
                    symbolBrush=pen.color(),
                    symbolSize=6,
                    )
            if(val["Plot 2"] and self.second_plot_ctrl.value()[0]):
                self.plot2.plot(
                    frames_visible,
                    signal_visible,
                    pen=pen,
                    name=name,
                    symbol=symbol,
                    symbolPen=pen,
                    symbolBrush=pen.color(),
                    symbolSize=6,
                    )
            iref +=1
         


    def _process_and_plot_tracking_signals(
        self,
        plot_widget1: pg.PlotWidget,
        plot_widget2: pg.PlotWidget,
        track_selection: dict[str, Any],
        sig_processing_vals: dict[str, Any],
        frame_begin: int,
        frame_end: int,
        show_points: bool,
        show_supports: bool,
    ) -> None:
        plot_widget1.clear()
        plot_widget2.clear()
        i1,i2 = 0,0


        IDs_to_process = []

        for key_ID, value_ID in track_selection.items():
            if key_ID == "Show":
                continue
            track_id = key_ID
            if value_ID is False:
                continue
            else:
                IDs_to_process.append(track_id)

        signalDict = {}
        for line_key, line_vals in sig_processing_vals.items():
            if line_vals["Method"] == "None":
                    continue
            
            lineSignals = {}
            for track_id in IDs_to_process:
                frames = self.tracking_data[track_id]["frames"]
                frames_mask = (frames >= frame_begin) & (frames <= frame_end)
                frames_visible = frames[frames_mask]
                if len(frames_visible) == 0:
                    continue

                signal_of_choice = line_vals["Input"]
                if signal_of_choice == "Ph. unwrap":
                    signal = self.tracking_data[track_id]["phase_unwrapped"]
                    signal_visible = signal[frames_mask]
                elif signal_of_choice in signalDict:
                    signal_visible = signalDict[signal_of_choice].get(track_id, np.array([]))
                else:
                    signal_visible = np.array([])
                    print(f"Invalid input {signal_of_choice} for {line_key}")


                
                if len(signal_visible) == 0:
                    continue

                signal_visible = self.signalProcessing(signal=signal_visible,method=line_vals["Method"],m_params=line_vals)
                lineSignals[track_id] = signal_visible
                

                if line_vals["Plot 1"] == False: # do not proceed to plotting if not to plot
                    pass
                else:
                    symbol = "x" if show_points else None
                    rcolor, g, b, acolor = self.cmap(i1 % 10)
                    color = (int(rcolor * 255), int(g * 255), int(b * 255), int(acolor * 255))
                    pen = pg.mkPen(color=color)
                    name = f"ID:{track_id} {line_key} - {line_vals['Method']}"

                    
                    
                    plot_widget1.plot(
                        frames_visible,
                        signal_visible,
                        pen=pen,
                        name=name,
                        symbol=symbol,
                        symbolPen=pen,
                        symbolBrush=pen.color(),
                        symbolSize=6,
                    )
                    i1 += 1

                    if show_supports:
                        support_y = np.max(signal_visible) * 1.05
                        disc_idx = np.where(np.diff(frames_visible) > 1)[0]
                        frame_segments = np.split(frames_visible, disc_idx)
                        for segment_frames in frame_segments:
                            if len(segment_frames) == 0:
                                continue
                            plot_widget1.plot(
                                segment_frames,
                                np.full(len(segment_frames), support_y),
                                pen=pen,
                            )
                            plot_widget1.plot(
                                [segment_frames[0], segment_frames[-1]],
                                [support_y, support_y],
                                pen=None,
                                symbol="|",
                                symbolPen=pen,
                                symbolBrush=pen.color(),
                                symbolSize=12,
                            )
                if line_vals["Plot 2"] == False or self.second_plot_ctrl.value()[0] == False : # do not proceed to plotting if not to plot
                    pass
                else:
                    symbol = "x" if show_points else None
                    rcolor, g, b, acolor = self.cmap(i2 % 10)
                    color = (int(rcolor * 255), int(g * 255), int(b * 255), int(acolor * 255))
                    pen = pg.mkPen(color=color)
                    name = f"ID:{track_id} {line_key} - {line_vals['Method']}"

                    
                    
                    plot_widget2.plot(
                        frames_visible,
                        signal_visible,
                        pen=pen,
                        name=name,
                        symbol=symbol,
                        symbolPen=pen,
                        symbolBrush=pen.color(),
                        symbolSize=6,
                    )
                    i2 += 1

                    if show_supports:
                        support_y = np.max(signal_visible) * 1.05
                        disc_idx = np.where(np.diff(frames_visible) > 1)[0]
                        frame_segments = np.split(frames_visible, disc_idx)
                        for segment_frames in frame_segments:
                            if len(segment_frames) == 0:
                                continue
                            plot_widget2.plot(
                                segment_frames,
                                np.full(len(segment_frames), support_y),
                                pen=pen,
                            )
                            plot_widget2.plot(
                                [segment_frames[0], segment_frames[-1]],
                                [support_y, support_y],
                                pen=None,
                                symbol="|",
                                symbolPen=pen,
                                symbolBrush=pen.color(),
                                symbolSize=12,
                            )
            
            signalDict[line_key] = lineSignals


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


        current_table_vals = self.sig_processing1_ctrl.value()
        last_methods = getattr(self.sig_processing1_ctrl, "previous_methods", None)
        selected_methods = [line_data.get("Method", "None") for line_data in current_table_vals.values()]
        if last_methods is None or selected_methods != last_methods:
            # The user changed a dropdown! We must rebuild the table schema.
            new_params = self.buildSignalControlDict(self.sig_processing1_ctrl.value(),lastMethods=self.sig_processing1_ctrl.previous_methods)
            self.sig_processing1_ctrl.set_newParams(params=new_params,silent=True)
            self.sig_processing1_ctrl.previous_methods = selected_methods # neew last methods are the previous methods now
        else:
            pass
        

        # print(self.sig_processing1_ctrl.value()) 

        self._update_second_plot_controls_state()

        
        self._process_and_plot_tracking_signals(
                self.plot,self.plot2,
                self.track_selection_ctrl.value(),
                self.sig_processing1_ctrl.value(),
                frame_begin,
                frame_end,
                show_points,
                show_supports,

            )
        
        self._plotHRs(frame_begin,frame_end)
            

        return
       

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

    def update_newParams_andHR(self, params, HRs):
        self.initDone = False
        
        self.params = params
        self.HRs = HRs

        self.frames_ctrl.set_range(params["i_Frames_begin"], params["i_Frames_end"] - 1)
        self.frames_ctrl.set_conv(lambda x: x * params["frame_index2time"])
        self.frames_ctrl.set_delta(params["frame_index2time"])

        paramsDict = self.buildRefSigControlDict()
        self.ref_signal_ctrl.set_newParams(paramsDict,headers=["PLOT 1","PLOT 2" ])

        self.initDone = True
        self.update_onSliderMove()


if __name__ == "__main__":
    app = QApplication([])
    window = PlotWindow()
    window.resize(800, 600)
    window.show()
    app.exec()
