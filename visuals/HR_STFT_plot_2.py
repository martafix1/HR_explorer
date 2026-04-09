from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QApplication, QGraphicsRectItem, QSplitter
)
from PySide6.QtGui import QPalette, QColor, QPen, QBrush
from PySide6.QtCore import Qt

import sys, types

import numpy as np
import scipy as sci
import pyqtgraph as pg
from superqt import QRangeSlider
import visuals.utils as utils
import visuals.param_controls as pctrl 

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


class HatchedBoundary():
    def __init__(self,pen :QPen ,plot_item : pg.PlotItem, angle ) -> None:
        self.pen = pen
        self.plot_item = plot_item
        self.angle = angle
        self.position = 0
       

        self.line = pg.InfiniteLine(pos=self.position, angle=self.angle%180, pen=pen)
        self.line.setZValue(5)
        
        self.rect = QGraphicsRectItem(0, 0, 1, 1)
        brush = QBrush(Qt.BDiagPattern)  # diagonal hatch
        brush.setColor(self.pen.color())
        self.rect.setBrush(brush)
        self.rect.setPen(QPen(Qt.NoPen))
        self.rect.setZValue(5)

        plot_item.addItem(self.line)
        plot_item.addItem(self.rect)
        
        pass

    def update_positions(self,position,x0,y0,x1,y1,dx,dy):
        self.position = position
        self.line.setPos(self.position)

        x_size = (x1-x0)
        y_size = (y1-y0)

        if self.angle == 0:
            self.rect.setRect(x0-dx,self.position,x_size+2*dx,-(self.position-y0)-dy)
        elif self.angle == 90:
            self.rect.setRect(self.position,y0-dy,(x1-self.position)+dx,y_size+2*dy)
        elif self.angle == 180:
            self.rect.setRect(x0-dx,self.position,x_size+2*dx,(y1 - self.position)+dy)
        elif self.angle == 270:
            self.rect.setRect(self.position,y0-dy,-(self.position-x0)-dx,y_size+2*dy)
        else:
            print("Unsupported rect angle")


class PlotWindow(QWidget):
    frames_ctrl             :   pctrl.RangeControl
    azi_ctrl                :   pctrl.SliderControl
    ele_ctrl                :   pctrl.SliderControl
    range_ctrl              :   pctrl.SliderControl
    stft_filters_ctrl       :   pctrl.RangeControl
    stft_filters_strenght_ctrl   :   pctrl.SliderControl
    stft_freqRange_ctrl     :   pctrl.RangeControl
    stft_winLen_ctrl        :   pctrl.SliderControl
    
    

    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ParamControl demo")
        self.resize(960, 580)

        self.initDone = False

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

        ctrl_panel = pctrl.ControlPanel(grid, callback=self.update_onSliderMove)

        # row 0 — frame range, cols: 0-2
        self.frames_ctrl = pctrl.RangeControl(
            "Frames selector", min_val=0, max_val=100,
            default=(15, 85), unit="s",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.frames_ctrl, row=0, col=0, col_span=3)

    
        # row 1 - Azimuth selector, col: 0
        self.azi_ctrl = pctrl.SliderControl(
            "Azimuth selector", min_val=0, max_val=100,
            default=2, unit="°",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.azi_ctrl, row=1, col=0)

        # row 1 - Elevation selector, col: 1
        self.ele_ctrl = pctrl.SliderControl(
            "Elevation selector", min_val=0, max_val=100,
            default=1, unit="°",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.ele_ctrl, row=1, col=1)


        # row 1 - range selector, col: 2
        self.range_ctrl = pctrl.SliderControl(
            "Range selector", min_val=0, max_val=100,
            default=10, unit="m",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.range_ctrl, row=1, col=2)

        # row 2 — Filter freq range, cols: 0-2
        self.stft_filters_ctrl = pctrl.RangeControl(
            "Filter freq selector", min_val=0, max_val=100,
            default=(0, 100), unit="Hz",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.stft_filters_ctrl, row=2, col=0, col_span=2)

        # row 2 - Filter strenght selector, col: 2
        self.stft_filters_strenght_ctrl = pctrl.SliderControl(
            "Filter strenght", min_val=0, max_val=32,
            default=0, unit="",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.stft_filters_strenght_ctrl, row=2, col=2)

        # row 3 — Shown frequencies range, cols: 0-2
        self.stft_freqRange_ctrl = pctrl.RangeControl(
            "Shown frequencies selector", min_val=0, max_val=100,
            default=(0, 100), unit="Hz",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.stft_freqRange_ctrl, row=3, col=0, col_span=2)

        # row 3 - range selector, col: 2
        self.stft_winLen_ctrl = pctrl.SliderControl(
            "STFT window lenght", min_val=0, max_val=100,
            default=50, unit="s",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.stft_winLen_ctrl, row=3, col=2)
       

        ctrl_widget.setMaximumHeight(80*4) # stuff is in control widget via the grid

        # ── plot ───────────────────────────────────────────────────────

        plot_widget = QWidget()
        plot_widget.setStyleSheet("background-color: #1e1e1e;")
        plot_layout = QVBoxLayout(plot_widget)
        
        # --------- PLOT ----------
        plot_item = pg.PlotItem()
        plot_item.showAxes(True)
        plot_item.setLabel('bottom', 'Time [s]')
        plot_item.setLabel('left', 'Frequency [Hz]')
        
        pen = QPen(pg.mkColor((255, 0, 0)))
        pen.setWidth(0.1)
        pen.setStyle(Qt.DashLine)


        pen.setColor(pg.mkColor((230, 120, 10)))
        self.overShootBegin_boundary = HatchedBoundary(pen,plot_item,270)
        self.overShootEnd_boundary = HatchedBoundary(pen,plot_item,90)
        pen.setColor(pg.mkColor((255, 0, 10)))
        self.validBegin_boundary = HatchedBoundary(pen,plot_item,270)
        self.validEnd_boundary = HatchedBoundary(pen,plot_item,90)

        pen.setColor(pg.mkColor((230, 230, 10)))
        self.LP_boundary = HatchedBoundary(pen,plot_item,0)
        self.HP_boundary = HatchedBoundary(pen,plot_item,180)


        # self.heatmap_validTime_begin_line.setZValue(10)
        # self.heatmap_validTime_end_line.setZValue(10)
        # # self.heatmap_LP_line.setZValue(10)
        # self.heatmap_HP_line.setZValue(10)


        self.heatmap = pg.ImageView(view=plot_item)
        colormap = pg.colormap.get("viridis")  # or "inferno", "plasma"
        self.heatmap.setColorMap(colormap)


        # Initial data
        data = np.random.rand(20,40)
        self.heatmap.setImage(data)

        plot_layout.addWidget(self.heatmap)

        # ── finnish splitter ───────────────────────────────────────────
        splitter.addWidget(ctrl_widget)
        splitter.addWidget(plot_widget)
        splitter.setSizes([220, 360])

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)
        
        self.initDone = True




    def update_onSliderMove(self, source : str = "None"):
        
        frame0,frame1 = self.frames_ctrl.value()
        azi_idx  = self.azi_ctrl.value()
        ele_idx = self.ele_ctrl.value()
        range_idx = self.range_ctrl.value()
        
        # todo: make this flash on range change
        self.stft_winLen_ctrl.set_range(0,int((frame1-frame0)/2))
        print(f"{(frame0-frame1)/2}, {frame0}, {frame1}")
        stft_winLen = self.stft_winLen_ctrl.value()

        
        stft_height = int(np.ceil((stft_winLen+1)/2))
        self.stft_filters_ctrl.set_range(0,stft_height)
        filter0,filter1 = self.stft_filters_ctrl.value()

        return
        if self.initDone == False:
            return
        
        

       
        # values updated bellow as soon as frequencies are known

        val_stft_LP_strenght = np.pow(2,self.stft_LP_strenght_slider.value()) 
        val_stft_HP_strenght = np.pow(2,self.stft_HP_strenght_slider.value()) 
        self.stft_LP_strenght_slider_label.setText(f"idx: {val_stft_LP_strenght :.2e} - 2^{self.stft_LP_strenght_slider.value()}")
        self.stft_HP_strenght_slider_label.setText(f"idx: {val_stft_HP_strenght :.2e} - 2^{self.stft_HP_strenght_slider.value()}")

        #freq range
        self.freq_range.setRange(0, stft_height)
        val_freq_begin,val_freq_end = self.freq_range.value()
        


        new_data = self.data4D[val_frame_begin:val_frame_end, val_range,val_ele,val_azi]

        # print(f"new_data shape: {new_data.shape}")

        window_len = val_stft_win
        win = sci.signal.windows.hann(np.max((1,window_len)) )
        STF =  sci.signal.ShortTimeFFT(win,hop = 2, fs = 1/self.params["frame_index2time"] )
        
        stftResult = np.abs(STF.stft(new_data)) 
        N_f, N_t = stftResult.shape
        # print(f"stft result (unrotated) shape: {stftResult.shape}")
        stftResult[0:val_stft_LP,:] /= val_stft_LP_strenght
        stftResult[val_stft_HP:0,:] /= val_stft_HP_strenght
        stftResult = stftResult[val_freq_begin:val_freq_end,:] # cut the freq a bit
        stftResult = np.rot90(stftResult, k=-1)

        t0,t1,f0,f1 = STF.extent(new_data.shape[0])
        dt = (t1-t0)/N_t
        df = (f1-f0)/N_f
        f1 -= ((N_f - val_freq_end)*df)
        f0 += (val_freq_begin*df)



        self.heatmap.getView().invertY(False)
        self.heatmap.setImage(stftResult,pos=(t0, f1), scale=(dt, -df))
        self.heatmap.getView().enableAutoRange(False)
        self.heatmap.getView().setAspectLocked(False)
        self.heatmap.getView().setRange(
            xRange=(t0, t1),
            yRange=(f0, f1),
            padding=0.02
        )
        
        # self.heatmap_validTime_end_line.setPos(t1)
        # self.heatmap_LP_line.setPos(val_stft_LP*df)
        # self.heatmap_HP_line.setPos(val_stft_HP*df)

        self.overShootBegin_boundary.update_positions(0-t0,t0,f0,t1,f1,dt,df)
        self.overShootEnd_boundary.update_positions(t1+2*t0,t0,f0,t1,f1,dt,df)

        self.validBegin_boundary.update_positions(0 ,t0,f0,t1,f1,dt,df)
        self.validEnd_boundary.update_positions(t1+(t0-0),t0,f0,t1,f1,dt,df)



        self.LP_boundary.update_positions(val_stft_LP*df,t0,f0,t1,f1,dt,df)
        self.HP_boundary.update_positions(val_stft_HP*df,t0,f0,t1,f1,dt,df)
        self.stft_LP_slider_label.setText(f"idx < {val_stft_LP} - f < {val_stft_LP*df} Hz")
        self.stft_HP_slider_label.setText(f"idx > {val_stft_HP} - f > {val_stft_HP*df} Hz ")
        self.freq_range_label.setText(f"idx: {val_freq_begin} - {val_freq_end}, ~ {val_freq_begin*df :.3f} - {val_freq_end*df :.3f} Hz")
        

        # t = np.linspace(frameTimes_s[0],frameTimes_s[1], val_frame_end-val_frame_begin)
        
            

    def update_newData(self,data,params):
        self.data4D = phaseUnwrapping(data)
        self.params = params

        self.frames_ctrl.set_range(params[ "i_Frames_begin"], params["i_Frames_end"]-1)
        self.frames_ctrl.set_conv(lambda x: float(x*self.params["frame_index2time"]))

        self.azi_ctrl.set_range(0, params["DoA_azi_N_elements"]-1)
        azi_index2deg_scale =  ((2*self.params["DoA_azi_range_degs"]) / self.params["DoA_azi_N_elements"])
        azi_index2deg_offset =  -1*self.params["DoA_azi_range_degs"] + azi_index2deg_scale/2
        self.azi_ctrl.set_conv(lambda x: float(x*azi_index2deg_scale + azi_index2deg_offset))

        self.ele_ctrl.set_range(0, params["DoA_ele_N_elements"]-1)
        ele_index2deg_scale =  ((2*self.params["DoA_ele_range_degs"]) / self.params["DoA_ele_N_elements"])
        ele_index2deg_offset =  -1*self.params["DoA_ele_range_degs"] + ele_index2deg_scale/2
        self.ele_ctrl.set_conv(lambda x: float(x*ele_index2deg_scale + ele_index2deg_offset))
        
        self.range_ctrl.set_range(0, params["i_Range_end"]-1)
        self.range_ctrl.set_conv(lambda x: float(x*self.params["range_index2dist"]))

        self.stft_winLen_ctrl.set_conv(lambda x: self.params["frame_index2time"]*x)
        
        self.stft_filters_strenght_ctrl.set_conv(lambda x: np.pow(2,x))
        self.stft_filters_strenght_ctrl._value_str = types.MethodType( # printing needs special attention
            lambda self: f"idx {self.value()} ~ {self._conv(self.value()):.2e}"
            if self._conv else f"idx {self.value()}",
            self.stft_filters_strenght_ctrl
        )
        # dataLen = params["i_Frames_end"]-params[ "i_Frames_begin"]
        # self.stft_window_slider.setValue(16)

        self.update_onSliderMove()
        pass



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())