from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QApplication, QGraphicsRectItem
)
from PySide6.QtGui import QPalette, QColor, QPen, QBrush
from PySide6.QtCore import Qt

import sys

import numpy as np
import scipy as sci
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
    def __init__(self):
        self.initDone = False
        super().__init__()


        self.setWindowTitle("STFT Window")
        self.resize(700, 500)

        main_layout = QVBoxLayout(self)

        # =========================
        # --- TOP CONTROL PANEL ---
        # =========================
        controls = QVBoxLayout()

        # --- Frame range (full width) ---
        self.frame_range = QRangeSlider(Qt.Horizontal) # type: ignore
        self.frame_range.setRange(0, 100)

        self.frame_range_label = QLabel()
        self.frame_range.valueChanged.connect(
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
        self._scrollWheel_filter = utils.SliderWheelFilter()
        def make_named_slider(attr_name, label_text, row, col):
            slider = QSlider(Qt.Horizontal) # type: ignore
            slider.setRange(0, 100)
            slider.installEventFilter(self._scrollWheel_filter)
            value_label = QLabel()

            slider.valueChanged.connect(
                self.update_onSliderMove
            )

            slider.setValue(0)
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

        # row 2
        self.stft_window_slider : QSlider
        self.stft_LP_slider : QSlider
        self.stft_HP_slider : QSlider

        self.stft_window_slider_label : QLabel
        self.stft_LP_slider_label : QLabel
        self.stft_HP_slider_label : QLabel

        make_named_slider("stft_window_slider", "Window lenght", 1, 0)
        # self.stft_window_slider.valueChanged.connect(lambda text, : self.update_onSliderMove("Window"))
        self.stft_window_slider.valueChanged.connect(self.update_onSliderMove)
        make_named_slider("stft_LP_slider", "Low pass", 1, 1)
        make_named_slider("stft_HP_slider", "High pass", 1, 2)
        self.stft_HP_slider.setValue(100)
        # row 3
        
        self.freq_range = QRangeSlider(Qt.Horizontal) # type: ignore
        self.freq_range.setRange(0, 100)

        self.freq_range_label = QLabel()
        self.freq_range.valueChanged.connect(
            self.update_onSliderMove
        )
        self.freq_range.setValue((10, 50))
        self.freq_range_label.setText("10 - 50")
        

        self.freq_range.setStyleSheet("""
        QRangeSlider {
            qproperty-barColor: #3ddc84;
        }
        """)

        # self.frame_range.setBarVisible(True)
       
        grid.addWidget(
            make_slider_block(
                "Freq start - Freq end",
                self.freq_range,
                self.freq_range_label
            ),2,0
        )


        self.stft_LP_strenght_slider : QSlider
        self.stft_HP_strenght_slider : QSlider

        self.stft_window_slider_label : QLabel
        self.stft_LP_strenght_slider_label : QLabel
        self.stft_HP_strenght_slider_label : QLabel

        make_named_slider("stft_LP_strenght_slider", "Low pass strenght", 2, 1)
        make_named_slider("stft_HP_strenght_slider", "High pass strenght", 2, 2)

        self.stft_LP_strenght_slider.setRange(0,32)
        self.stft_HP_strenght_slider.setRange(0,32)

        controls.addLayout(grid)

        # limit total height of control area
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumHeight(180)

        main_layout.addWidget(controls_widget)


        # --------- PLOT ----------
        plot_item = pg.PlotItem()
        plot_item.showAxes(True)
        plot_item.setLabel('bottom', 'Time [s]')
        plot_item.setLabel('left', 'Frequency [Hz]')
        
        pen = QPen(pg.mkColor((255, 0, 0)))
        pen.setWidth(0.1)
        pen.setStyle(Qt.DashLine)

        # self.heatmap_validTime_begin_line = pg.InfiniteLine(pos=0, angle=90, pen=pen)
        # self.heatmap_validTime_end_line =   pg.InfiniteLine(pos=0, angle=90, pen=pen)
        
        # self.heatmap_LP_line =   pg.InfiniteLine(pos=0, angle=0, pen=pen)
        # self.heatmap_HP_line =   pg.InfiniteLine(pos=0, angle=0, pen=pen)
        # plot_item.addItem(self.heatmap_validTime_begin_line)
        # plot_item.addItem(self.heatmap_validTime_end_line)
        # plot_item.addItem(self.heatmap_LP_line)
        # plot_item.addItem(self.heatmap_HP_line)
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
        # Optional: nicer color map
        colormap = pg.colormap.get("viridis")  # or "inferno", "plasma"
        self.heatmap.setColorMap(colormap)


        # Initial data
        data = np.random.rand(20,40)
        self.heatmap.setImage(data)


        main_layout.addWidget(self.heatmap)
        self.initDone = True



    def update_onSliderMove(self, source : str = "None"):
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
        

        self.stft_window_slider.setRange(0,(val_frame_end-val_frame_begin)/2)
        val_stft_win =  self.stft_window_slider.value()
        self.stft_window_slider_label.setText(f"idx: {val_stft_win} - {val_stft_win*self.params["frame_index2time"] :.2f} s ")
        # set ranges for filters
        stft_height = int(np.ceil((val_stft_win+1)/2)) 
        self.stft_HP_slider.setRange(0,stft_height)
        self.stft_LP_slider.setRange(0,stft_height)

        val_stft_LP =  self.stft_LP_slider.value()
        
        val_stft_HP =  self.stft_HP_slider.value()
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

        self.frame_range.setRange(params[ "i_Frames_begin"], params["i_Frames_end"]-1)
        self.azi_slider.setRange(0, params["DoA_azi_N_elements"]-1)
        self.ele_slider.setRange(0, params["DoA_ele_N_elements"]-1)
        self.range_slider.setRange(0, params["i_Range_end"]-1)

        dataLen = params["i_Frames_end"]-params[ "i_Frames_begin"]
        self.stft_window_slider.setValue(16)

        self.update_onSliderMove()
        pass



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())