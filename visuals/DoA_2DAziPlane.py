import sys
import numpy as np
from PySide6 import QtWidgets, QtGui
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QApplication, QGraphicsRectItem, QSplitter
)

from PySide6.QtCore import Qt, QTimer, Signal


import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import visuals.utils
import visuals.param_controls as pctrl 



def aggregate(data :np.ndarray, axis :int, ag_type:str) -> np.ndarray :
    ops = {
        "sum": np.sum,
        "mean": np.mean,
        "max": np.max
    }
    return ops[ag_type.lower()](a=data,axis=axis)



class PlotWindow(QWidget):
    frames_ctrl              :   pctrl.RangeControl
    frames_aggregation_ctrl  :   pctrl.ComboControl
    range_ctrl               :   pctrl.RangeControl
    doppler_ctrl             :   pctrl.RangeControl
    doppler_aggregation_ctrl :   pctrl.ComboControl
    

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

        # row 0 — frame range, cols: 0-1
        self.frames_ctrl = pctrl.RangeControl(
            "Frames selector", min_val=0, max_val=100,
            default=(15, 85), unit="s",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.frames_ctrl, row=0, col=0, col_span=2)

        # row 0 - frame aggregation selector, col: 2
        self.frames_aggregation_ctrl = pctrl.ComboControl(
                "Frames aggregation selector",
                options=["Mean", "Sum" , "Max"],
                default=0,
            )
        ctrl_panel.add(self.frames_aggregation_ctrl, row=0, col=2)

        # row 1 - range selector, cols: 0-3
        self.range_ctrl = pctrl.RangeControl(
            "Range selector", min_val=0, max_val=100,
            default=(5, 50), unit="m",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.range_ctrl, row=1, col=0, col_span=3)

        # row 2 — doppler range, cols: 0-1
        self.doppler_ctrl = pctrl.RangeControl(
            "Doppler selector", min_val=0, max_val=100,
            default=(0, 100), unit="Hz",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.doppler_ctrl, row=2, col=0, col_span=2)

        # row 2 - frame aggregation selector, col: 2
        self.doppler_aggregation_ctrl = pctrl.ComboControl(
                "Doppler aggregation selector",
                options=["Mean", "Sum" , "Max"],
                default=0,
            )
        ctrl_panel.add(self.doppler_aggregation_ctrl, row=2, col=2)

        ctrl_widget.setMaximumHeight(80*2) # stuff is in control widget via the grid

        # ── plot ───────────────────────────────────────────────────────
        plot_widget = QWidget()
        plot_widget.setStyleSheet("background-color: #1e1e1e;")
        plot_layout = QVBoxLayout(plot_widget)
        aziplot =  pg.GraphicsLayoutWidget()
        self.plot_item = aziplot.addPlot()
        R, T = np.meshgrid(np.array([0,1,2]), np.array([0,1,2]), indexing='ij')
        self.mesh = pg.PColorMeshItem(
            R, T, np.random.rand(2,2),
            colorMap=pg.colormap.get("inferno"),
            edgecolors=None,
            #edgecolors=(50, 50, 200),
            antialiasing=False
        )
        self.plot_item.addItem(self.mesh)

        self.cbar = pg.ColorBarItem(
            colorMap=pg.colormap.get("inferno"),
            values=(0, 1)
        )
        self.cbar.setImageItem(self.mesh) #link to mesh to control the colors
        aziplot.addItem(self.cbar)
        self.PolarAxisItems = []

        plot_layout.addWidget(aziplot)


        # ── finnish splitter ───────────────────────────────────────────
        splitter.addWidget(ctrl_widget)
        splitter.addWidget(plot_widget)
        splitter.setSizes([220, 360])

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)
        
        self.initDone = True
        # self.update_onSliderMove()
        # self.on_change()   # initial draw

    def _drawPolarAxes(self,r0 = 0, r1=2,phi0 = -90,phi1 = 90, dr = 0.5, dphi = 45):
        #def draw_polar_grid(self, plot_item, r_max, r_steps=4, theta_steps=8):
        # clear old items
        for item in self.PolarAxisItems:
            self.plot_item.removeItem(item)
        self.PolarAxisItems.clear()

        # --- circles (radius) ---
        r_values = np.arange(r0, r1 + dr/2, dr)
        r_values = np.append(r_values, r1)
        # r_values = np.append(r_values, r0)
        for r in r_values:
            path = QtGui.QPainterPath()
            path.arcMoveTo(-r, -r, 2*r, 2*r, phi0 -90)
            path.arcTo(-r, -r, 2*r, 2*r, phi0 -90, (phi1-phi0))

            arc_item = QtWidgets.QGraphicsPathItem(path)
            arc_item.setPen(pg.mkPen((150, 150, 150, 100)))
            self.plot_item.addItem(arc_item)
            self.PolarAxisItems.append(arc_item)

            # --- radius label ---
            label = pg.TextItem(f"{r:.2f}", anchor=(0, 0.5))
            
            # small horizontal offset so it’s not on top of the axis
            label.setPos(0.05 * dr, r)

            self.plot_item.addItem(label)
            self.PolarAxisItems.append(label)

        # --- radial lines (angles) ---
        #this would work if it were linspace :) need 1/2 dphi to be sure that the phi1 fits in and then need another one coz the dphi is to count the values and the edges are one more.
        # theta_values = np.deg2rad(np.arange(phi0, phi1 + dphi*1/2, dphi))
        # +2 coz +1 to fit in the last value and +1 coz we want edges not centers
        theta_values = np.linspace(phi0,phi1, int(np.ceil((phi1-phi0)/dphi)) +2  )
        for theta in np.deg2rad(theta_values):
            x = r1 * np.sin(theta) #not normal in order to keep y positive - facing up
            y = r1 * np.cos(theta)
            line = QtWidgets.QGraphicsLineItem(0, 0, x, y)
            line.setPen(pg.mkPen((150, 150, 150, 100)))
            self.plot_item.addItem(line)
            self.PolarAxisItems.append(line)

            # --- label ---
            label = pg.TextItem(f"{np.rad2deg(theta) :.0f}°", anchor=(0.5, 0.5))
            
            # push slightly outward so it doesn't sit on the line end
            offset = 1.1
            label.setPos(offset * x, offset * y)

            self.plot_item.addItem(label)
            self.PolarAxisItems.append(label)


    def update_onSliderMove(self):
        if self.initDone == False:
            return
        
        frame0,frame1 = self.frames_ctrl.value()
        doppler0,doppler1 = self.doppler_ctrl.value()
        range0,range1 = self.range_ctrl.value()
        new_data = np.abs(self.penteract[frame0:frame1,doppler0:doppler1,range0:range1,0,:]) 
        new_data = aggregate(new_data,axis=0,ag_type= self.frames_aggregation_ctrl.value())
        #axis 0 is now doppler:
        new_data = aggregate(new_data,axis=0, ag_type= self.doppler_aggregation_ctrl.value())

        # plotting:
        ## upscale:
        N_azi = new_data.shape[1]
        scale_azi = max(1, int(np.ceil(16 / N_azi)))
        new_data = np.repeat(new_data, scale_azi, axis=1)
        azi_points = self.params["azi_points"]
        if azi_points.shape[0] < 2:
            azi_points = np.array([azi_points[0],-azi_points[0]]) 
        og_azi_angle_diff = azi_points[1]-azi_points[0]
        azi_points = np.linspace( azi_points[0], azi_points[-1], new_data.shape[1]+1)

        range0_m = range0 * self.params["range_index2dist"]
        range1_m = range1 * self.params["range_index2dist"]
        r_edges = np.linspace(range0_m, range1_m, (range1-range0) + 1)
        

        R, T = np.meshgrid(r_edges, np.deg2rad(azi_points) , indexing='ij')
        
        # facing UP
        X = R * np.sin(T)
        Y = R * np.cos(T)

        # Data
        Z = new_data


        self.mesh.setData(X,Y,Z) 

        self._drawPolarAxes(range0_m,range1_m,azi_points[0], azi_points[-1], 2 * self.params["range_index2dist"],og_azi_angle_diff )
        
        pass

    def update_newData(self,data,params):
        self.penteract = data
        self.params = params

        
        self.frames_ctrl.set_range(params[ "i_Frames_begin"], params["i_Frames_end"]-1)
        self.frames_ctrl.set_conv(lambda x: float(params["frame_index2time"]*x) )
        self.frames_ctrl.set_delta(params["frame_index2time"])

        self.range_ctrl.set_range( params["i_Range_begin"], params["i_Range_end"]-1)
        self.range_ctrl.set_conv(lambda x: float(params["range_index2dist"]*x) )
        
        self.doppler_ctrl.set_range(params["i_Doppler_begin"],params["i_Doppler_end"]-1 )
        self.doppler_ctrl.set_conv(lambda x: float(params["doppler_index2freq"]*x) )

        if params["Doppler_processing"] == "None":
            self.doppler_ctrl.set_warning("Data is not doppler processed!")
            self.doppler_aggregation_ctrl.set_warning("Data is not doppler processed!")
        else:
            self.doppler_ctrl.clear_state()
            self.doppler_aggregation_ctrl.clear_state()

        self.update_onSliderMove()
        pass


class RadarPlot(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Radar Wedge Plot (Polar Grid)")

        self.plot = self.addPlot()
        self.plot.setAspectLocked()
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')

        # Parameters
        self.n_r = 120
        self.n_theta = 8
        self.r_max = 100

        self.theta_min = -np.pi / 3
        self.theta_max = np.pi / 3

        # Bin edges
        r_edges = np.linspace(0, self.r_max, self.n_r + 1)
        theta_edges = np.linspace(self.theta_min, self.theta_max, self.n_theta + 1)

        R, T = np.meshgrid(r_edges, theta_edges, indexing='ij')

        # ✅ Correct orientation (facing UP)
        X = R * np.sin(T)
        Y = R * np.cos(T)

        # Data
        Z = np.random.rand(self.n_r, self.n_theta)

        cmap = pg.colormap.get("viridis")

        self.mesh = pg.PColorMeshItem(
            X, Y, Z,
            colorMap=cmap,
            edgecolors=None,
            #edgecolors=(50, 50, 200),
            antialiasing=False
        )

        self.plot.addItem(self.mesh)

        # Add grid
        self.add_polar_grid()

        # Set view so origin is at bottom
        # self.plot.setLimits(xMin=X.min(), xMax=X.max(), yMin=0, yMax=Y.max())
        self.plot.setRange(xRange=(X.min(), X.max()), yRange=(0, Y.max()))

    def add_polar_grid(self):
        pen = pg.mkPen((200, 200, 200, 200), width=2)

        # --- Range rings ---
        for r in np.linspace(20, self.r_max, 5):
            circle = QtWidgets.QGraphicsEllipseItem(-r, 0, 2*r, 2*r)
            circle.setPen(pen)
            self.plot.addItem(circle)

        # --- Angle lines ---
        for theta in np.linspace(self.theta_min, self.theta_max, 7):
            x = self.r_max * np.sin(theta)
            y = self.r_max * np.cos(theta)

            line = pg.PlotDataItem([0, x], [0, y], pen=pen)
            self.plot.addItem(line)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setCentralWidget(RadarPlot())
        self.setCentralWidget(PlotWindow())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())