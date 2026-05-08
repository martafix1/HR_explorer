import sys
import numpy as np
from PySide6 import QtWidgets, QtGui
import pyqtgraph as pg
import random

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QApplication, QGraphicsRectItem, QSplitter
)

from PySide6.QtCore import Qt, QTimer, Signal


import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import visuals.utils
import visuals.param_controls as pctrl 
import processing.cfar as cfar
import processing.tracking as tracking

import matplotlib


def aggregate(data :np.ndarray, axis :int, ag_type:str) -> np.ndarray :
    ops = {
        "sum": np.sum,
        "mean": np.mean,
        "max": np.max
    }
    return ops[ag_type.lower()](a=data,axis=axis)



class PlotWindow(QWidget):
    frames_ctrl              :   pctrl.SliderControl
    range_ctrl               :   pctrl.RangeControl
    cfar_table_ctrl          :   pctrl.DictTableControl
    track_btn_ctrl           :   pctrl.ButtonControl

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
        self.frames_ctrl = pctrl.SliderControl(
            "Frames selector", min_val=0, max_val=100,
            default=(15), unit="s",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.frames_ctrl, row=0, col=0, col_span=3)

        
        # row 1 - range selector, cols: 0-3
        self.range_ctrl = pctrl.RangeControl(
            "Range selector", min_val=0, max_val=128,
            default=(6, 30), unit="m",
            conv=lambda x: float(x),
        )
        ctrl_panel.add(self.range_ctrl, row=1, col=0, col_span=2)

        self.track_btn_ctrl = pctrl.ButtonControl("Tracking", button_label="Run Tracker")
        self.track_btn_ctrl._btn.clicked.connect(self.runTrackingLogic)

        ctrl_panel.add(self.track_btn_ctrl,row=1,col=2)


        self.cfar_table_ctrl = pctrl.DictTableControl(
                "CFAR params",
                params={
                    "Show": ["targets","treshold","none"],
                    "Method": ["cMax 2D CFAR", "CA 2D CFAR"],
                    "guard range": 1,
                    "train range": 2,
                    "guard azi": 0,
                    "train azi": 1,
                    "treshold scale": 1.0,
                    "dopp0 weight": 1.0,
                    "dopp>0 weight": 1.0
                },
                orientation="horizontal",
            )
        ctrl_panel.add(self.cfar_table_ctrl, row=2, col=0,col_span=3)
        
        self.tracking_table_ctrl = pctrl.DictTableControl(
                "Tracking params",
                params={
                    #"Show": ["targets","treshold","none"],
                    "match W_range": 1.0,
                    "match W_azi": 2.0,
                    "gating dist": 3.0,
                    "show history": 2,
                    "Sig. extr: merge bins": True,
                    "Clustering": ["4 way NN ampl. mean"],
                    "Cost assignment": ["weighted norm 2"],
                    "Assignment": ["Gated Glob.NN"],
                },
                orientation="vertical",
            )
        ctrl_panel.add(self.tracking_table_ctrl, row=0, col=4,row_span=3)

       
        
        ctrl_widget.setMaximumHeight(80*3) # stuff is in control widget via the grid

        # ── plot ───────────────────────────────────────────────────────
        plot_widget = QWidget()
        plot_widget.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
        }

        QToolTip {
            background-color: yellow;
            color: black;
            border: 1px solid black;
        }
        """)
        plot_layout = QVBoxLayout(plot_widget)
        aziplot =  pg.GraphicsLayoutWidget()
        self.plot_item = aziplot.addPlot()
        R, T = np.meshgrid(np.array([0,1,2]), np.array([0,1,2]), indexing='ij')
        colormap_ = "magma" # "inferno", "plasma", "viridis"
        self.mesh = pg.PColorMeshItem(
            R, T, np.random.rand(2,2),
            colorMap=pg.colormap.get(colormap_),
            edgecolors=None,
            #edgecolors=(50, 50, 200),
            antialiasing=False
        )
        self.plot_item.addItem(self.mesh)

        self.cbar = pg.ColorBarItem(
            colorMap=pg.colormap.get(colormap_),
            values=(0, 1)
        )
        self.cbar.setImageItem(self.mesh) #link to mesh to control the colors
        aziplot.addItem(self.cbar)
        self.PolarAxisItems = []

        plot_layout.addWidget(aziplot)

        # ── tracker overlay ────
        self.track_labels = []
        self.dispIDs = True
        self.track_death_markers = []

        # Precompute a palette 
        self.track_color_pool = [
            (255, 80, 80),    # bright red
            (80, 255, 80),    # bright green
            (80, 160, 255),   # bright blue
            (255, 200, 80),   # orange
            (255, 80, 200),   # magenta
            (80, 255, 255),   # cyan
            (255, 255, 80),   # yellow
            (180, 120, 255),  # light purple
            (255, 140, 140),  # soft red
            (140, 255, 200),  # mint
        ]

        self.track_color_map = {}   # tid -> color
        self.track_active_ids = set()

        # This scatter plot handles the "heads" of the tracks
        self.track_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('w'), brush=pg.mkBrush('r'), 
            hoverable=True, tip=self._format_track_tooltip
        )
        # This list will hold the "tails" (history lines)
        self.track_tails = [] 
        self.plot_item.addItem(self.track_scatter)

        # ── finnish splitter ───────────────────────────────────────────
        splitter.addWidget(ctrl_widget)
        splitter.addWidget(plot_widget)
        splitter.setSizes([220, 360])

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)
        
        self.initDone = True
        # self.update_onSliderMove()
        # self.on_change()   # initial draw
 # the r0,1 and phi0,1 shall be edges, not centers
    def _drawPolarAxes(self,r0 = 0, r1=2,phi0 = -90,phi1 = 90, dr = 0.5, dphi = 45,skip_rangeLabels = 2):
        #def draw_polar_grid(self, plot_item, r_max, r_steps=4, theta_steps=8):
        # clear old items
        for item in self.PolarAxisItems:
            self.plot_item.removeItem(item)
        self.PolarAxisItems.clear()

        # --- circles (radius) ---
        r_values = np.arange(r0, r1 + dr/2, dr)
        r_values = np.append(r_values, r1)
        # r_values = np.append(r_values, r0)
        for i,r in enumerate(r_values):
            path = QtGui.QPainterPath()
            path.arcMoveTo(-r, -r, 2*r, 2*r, phi0 -90)
            path.arcTo(-r, -r, 2*r, 2*r, phi0 -90, (phi1-phi0))

            arc_item = QtWidgets.QGraphicsPathItem(path)
            if i%skip_rangeLabels == 0:
                arc_item.setPen(pg.mkPen((150, 150, 150, 100)))
            else:
                arc_item.setPen(pg.mkPen((150, 150, 150, 50)))
            self.plot_item.addItem(arc_item)
            self.PolarAxisItems.append(arc_item)

            # --- radius label ---
            if i%skip_rangeLabels == 0:
                center_r = r + dr/2
                label = pg.TextItem(f"{center_r:.2f}", anchor=(0, 0.5))
            
                # small horizontal offset so it’s not on top of the axis
                label.setPos(0.1 * dr, center_r)

                self.plot_item.addItem(label)
                self.PolarAxisItems.append(label)

        # --- radial lines (angles) ---        
        theta_values = np.arange(phi0, phi1 + dphi*1/2, dphi/2)
        theta_values_centers = np.arange(phi0, phi1, dphi) + dphi/2

        for i,theta in enumerate(np.deg2rad(theta_values)):
            if i%2 == 0: # edge
                r = r1 + 2*dr
                pen = pg.mkPen((150, 150, 150, 100))
            else: # center
                r = r1 

            x = r * np.sin(theta) #not normal in order to keep y positive - facing up
            y = r * np.cos(theta)
            if i%2==0:
                line = QtWidgets.QGraphicsLineItem(0, 0, x, y)
                pen = pg.mkPen((150, 150, 150, 100))
                line.setPen(pen)
                self.plot_item.addItem(line)
                self.PolarAxisItems.append(line)

            # --- label ---
            label = pg.TextItem(f"{np.rad2deg(theta) :.0f}°", anchor=(0.5, 0.5))
            
            # push slightly outward so it doesn't sit on the line end
            offset = 1.1
            label.setPos(offset * x, offset * y)

            self.plot_item.addItem(label)
            self.PolarAxisItems.append(label)

    def _format_track_tooltip(self,x, y, data):
            # 'data' will now be your entire track dictionary
            if isinstance(data, dict):
                # Dynamically create a list of "Key: Value" strings
                lines = [f"<b>{key}</b>: {value}" for key, value in data.items()]
                # Join them with HTML line breaks
                return "<br>".join(lines)
            
            # Fallback just in case
            return str(data)

    def _update_track_overlay(self):
        if not hasattr(self, 'tracking_history'):
            return
        historyDepth = self.tracking_table_ctrl.value()["show history"]
        # --- A. Clear previous visuals ---
        for tail in self.track_tails:
            self.plot_item.removeItem(tail)
        self.track_tails = []

        for label in self.track_labels:
            self.plot_item.removeItem(label)
        self.track_labels = []

        for marker in self.track_death_markers:
            self.plot_item.removeItem(marker)
        self.track_death_markers = []

        # --- B. Current frame ---
        current_rel_frame = self.frames_ctrl.value() - self.frames_ctrl._min

        range0,range1 = self.range_ctrl.value()

        # --- C. Build track histories ---
        track_paths = {}
        current_ids = set()

        d_azi = self.params["azi_points"][1]-self.params["azi_points"][0] # for angle calcs

        for frame_idx in range(max(0, current_rel_frame - historyDepth), current_rel_frame + 1):
            frame_data = self.tracking_history[frame_idx]

            for t in frame_data:
                range_abs = t['r_bin'] + self.trackingRangeOffset 
                # if range_abs > range1 or range_abs < range0:
                #     continue

                tid = t['id']
                current_ids.add(tid)
                if "centroid" in t: #supports exact position
                    r_phys = (t["centroid"][0] + self.trackingRangeOffset  ) * self.params["range_index2dist"]
                    
                    azi_angle = t["centroid"][1] * d_azi + self.params["azi_points"][0]
                else: # grid only position    
                    r_phys = (range_abs ) * self.params["range_index2dist"]
                    azi_angle = self.params["azi_points"][t['azi_bin']]

                x = r_phys * np.sin(np.deg2rad(azi_angle))
                y = r_phys * np.cos(np.deg2rad(azi_angle))

                track_paths.setdefault(tid, []).append((x, y))

        # --- D. Maintain color pool (assign + release) ---
        # release colors of dead tracks
        dead_ids = self.track_active_ids - current_ids
        for tid in dead_ids:
            if tid in self.track_color_map:
                self.track_color_pool.append(self.track_color_map.pop(tid))

        # assign colors to new tracks
        new_ids = current_ids - self.track_active_ids
        for tid in new_ids:
            if self.track_color_pool:
                self.track_color_map[tid] = self.track_color_pool.pop(0)
            else:
                # fallback (shouldn't happen with ≤10 tracks)
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)
                self.track_color_map[tid] = (r, g, b)

        self.track_active_ids = current_ids.copy()

        # --- E. Detect deaths ---
        death_positions = []

        for frame_idx in range(max(0, current_rel_frame - historyDepth), current_rel_frame):
            ids_now = {t['id'] for t in self.tracking_history[frame_idx]}
            ids_next = {t['id'] for t in self.tracking_history[frame_idx + 1]}

            died = ids_now - ids_next

            for t in self.tracking_history[frame_idx]:
                if t['id'] in died:
                    range_abs = t['r_bin'] + self.trackingRangeOffset 
                    if range_abs > range1 or range_abs < range0:
                        continue
                    
                    if "centroid" in t: #supports exact position
                        r_phys = (t["centroid"][0] + self.trackingRangeOffset  ) * self.params["range_index2dist"]
                        
                        azi_angle = t["centroid"][1] * d_azi + self.params["azi_points"][0]
                    else: # grid only position    
                        r_phys = (range_abs ) * self.params["range_index2dist"]
                        azi_angle = self.params["azi_points"][t['azi_bin']]

                    x = r_phys * np.sin(np.deg2rad(azi_angle))
                    y = r_phys * np.cos(np.deg2rad(azi_angle))

                    death_positions.append((t['id'], x, y))

        # --- F. Draw tails with fading ---
        for tid, path in track_paths.items():
            if len(path) < 2:
                continue

            color = self.track_color_map.get(tid, (255, 255, 255))

            # draw segments with fading alpha
            for i in range(len(path) - 1):
                (x1, y1), (x2, y2) = path[i], path[i + 1]

                alpha = int(255 * (i + 1) / len(path))  # fade in toward present

                seg = pg.PlotDataItem(
                    x=[x1, x2],
                    y=[y1, y2],
                    pen=pg.mkPen(color=(*color, alpha), width=2)
                )

                self.plot_item.addItem(seg)
                self.track_tails.append(seg)

            # --- Label ---
            if self.dispIDs:
                x_last, y_last = path[-1]

                label = pg.TextItem(
                    text=str(tid),
                    color=color,
                    anchor=(0, 1)
                )
                label.setPos(x_last, y_last)

                self.plot_item.addItem(label)
                self.track_labels.append(label)

        # --- G. Death markers ---
        for tid, x, y in death_positions:
            color = self.track_color_map.get(tid, (0, 255, 0))

            marker = pg.ScatterPlotItem(
                x=[x],
                y=[y],
                symbol='x',
                size=12,
                pen=pg.mkPen(color, width=2)
            )

            self.plot_item.addItem(marker)
            self.track_death_markers.append(marker)

        # --- H. Heads ---
        current_frame_data = self.tracking_history[current_rel_frame]

        points = []
        brushes = []

        for t in current_frame_data:
            tid = t['id']
            range_abs = t['r_bin'] + self.trackingRangeOffset 
            if range_abs > range1 or range_abs < range0:
                continue
            color = self.track_color_map.get(tid, (255, 255, 255))

            if "centroid" in t: #supports exact position
                r_phys = (t["centroid"][0] + self.trackingRangeOffset  ) * self.params["range_index2dist"]
                azi_angle = t["centroid"][1] * d_azi + self.params["azi_points"][0]
            else: # grid only position    
                r_phys = (range_abs ) * self.params["range_index2dist"]
                azi_angle = self.params["azi_points"][t['azi_bin']]

            x = r_phys * np.sin(np.deg2rad(azi_angle))
            y = r_phys * np.cos(np.deg2rad(azi_angle))

            points.append({'pos': (x, y), 'data': t})
            brushes.append(pg.mkBrush(color))

        self.track_scatter.setData(points, brush=brushes)

    def runTrackingLogic_placeholder(self):
        # A. Determine number of frames from slider
        f0, f1 = self.params[ "i_Frames_begin"], self.params["i_Frames_end"]
        num_frames = f1 - f0

        # Grid size (as requested)
        R_MAX = 20   # range bins
        A_MAX = 8    # azimuth bins

        # B. Create empty track history (list of frames)
        track_history = [[] for _ in range(num_frames)]

        # C. Generate random number of tracks
        num_tracks = random.randint(10, 30)

        for track_id in range(num_tracks):

            # Random start and lifetime
            start_frame = random.randint(0, max(0, num_frames - 1))
            lifetime = random.randint(1, 10)

            # Initial position
            r = random.randint(0, R_MAX - 1)
            a = random.randint(0, A_MAX - 1)

            for step in range(lifetime):
                frame_idx = start_frame + step
                if frame_idx >= num_frames:
                    break

                # Store current position
                track_history[frame_idx].append({
                    'id': track_id,
                    'r_bin': r,
                    'azi_bin': a
                })

                # Random walk step (-1, 0, +1)
                dr = random.choice([-1, 0, 1])
                da = random.choice([-1, 0, 1])

                # Update position with boundary clamp
                r = max(0, min(R_MAX - 1, r + dr))
                a = max(0, min(A_MAX - 1, a + da))

        # D. Store and visualize
        self.tracking_history = track_history
        self._update_track_overlay()

    def runTrackingLogic(self):
        range0,range1 = self.range_ctrl.value()
        self.trackingRangeOffset = range0
        # self.tracking_table_ctrl.value()
        self.tracking_history, self.track_signals = tracking.track_allData(penteract= self.penteract[:,:,range0:range1,:,:],
                                                       cfar_params=self.cfar_table_ctrl.value(),
                                                       tracking_params=self.tracking_table_ctrl.value())
        self._update_track_overlay()

    def returnTrackedSignals(self):
        if not hasattr(self, 'track_signals'):
            return {}
        
        return self.track_signals

    def update_onSliderMove(self):
        if self.initDone == False:
            return
        
        frame0  = self.frames_ctrl.value()
        frame0 -= self.params[ "i_Frames_begin"]
        what2show, method, g_range,t_range, g_azi,t_azi,treshold_scale, dopp0_w, doppHigh_w =  self.cfar_table_ctrl.value_tuple()
        


        range0,range1 = self.range_ctrl.value()
        new_data = np.abs(self.penteract[frame0,:,range0:range1,0,:]) 
        #axis 0 is now doppler:
        new_data[0,:,:] *= dopp0_w
        new_data[1:,:,:] *=  doppHigh_w
        new_data = aggregate(new_data,axis=0, ag_type= "Mean")


        #CFAR
        if what2show == "none":
            pass
        else:
            if method == "CA 2D CFAR":
                treshold = cfar.cell_average_CFAR_2D(new_data,g_range,g_azi,t_range,t_azi,dim0_i=0,dim1_i=1)
            elif method == "cMax 2D CFAR":
                treshold = cfar.cell_max_CFAR_2D(new_data,g_range,g_azi,t_range,t_azi)
            else:
                treshold = new_data  
                print("Unknown CFAR method")
            
            treshold *=  treshold_scale # "treshold scale"
            if what2show == "treshold":
                new_data = treshold
            elif what2show == "targets":
                targets = new_data
                targets[new_data<treshold] = 0
                new_data = targets
        

    # plotting:
        ## upscale:
        N_azi = new_data.shape[1]
        scale_azi = max(1, int(np.ceil(16 / N_azi)))
        new_data = np.repeat(new_data, scale_azi, axis=1)
        azi_beamVectors = self.params["azi_points"]
        if azi_beamVectors.shape[0] < 2:
            azi_beamVectors = np.array([azi_beamVectors[0],-azi_beamVectors[0]]) 
        azi_diff = azi_beamVectors[1]-azi_beamVectors[0]
        azi_edges = azi_beamVectors - azi_diff/2 
        azi_edges = np.append(azi_edges,(azi_edges[-1]+azi_diff))
        og_azi_angle_diff = azi_diff
        # azi_points = np.linspace( azi_points[0], azi_points[-1], new_data.shape[1]+1)
        azi_edges_scaled =  np.linspace( azi_edges[0], azi_edges[-1], new_data.shape[1]+1) # azi_beamVectors
        
        range0_m = range0 * self.params["range_index2dist"]
        range1_m = range1 * self.params["range_index2dist"]
        r_edges = np.linspace(range0_m, range1_m, (range1-range0) + 1) - self.params["range_index2dist"]/2
        
        # PColorMeshItem plots data centers (Z) limited by the edges (X,Y), edges must be one biggur
        R, T = np.meshgrid(r_edges, np.deg2rad(azi_edges_scaled) , indexing='ij')
        
        # facing UP
        X = R * np.sin(T)
        Y = R * np.cos(T)

        # Data
        Z = new_data


        self.mesh.setData(X,Y,Z) 

        self._drawPolarAxes(r_edges[0],r_edges[-1],
                            phi0= azi_edges[0],
                            phi1= azi_edges[-1],
                            dr= self.params["range_index2dist"],
                            dphi=og_azi_angle_diff,
                            skip_rangeLabels= 2 )
        
        # trackers overlay
        self._update_track_overlay()

        pass

    def update_newData(self,data,params):
        self.penteract = data
        self.params = params

        
        self.frames_ctrl.set_range(params[ "i_Frames_begin"], params["i_Frames_end"]-1)
        self.frames_ctrl.set_conv(lambda x: float(params["frame_index2time"]*x) )
        self.frames_ctrl.set_delta(params["frame_index2time"])

        self.range_ctrl.set_range( params["i_Range_begin"], params["i_Range_end"]-1)
        self.range_ctrl.set_conv(lambda x: float(params["range_index2dist"]*x) )
        


        if params["Doppler_processing"] == "None":
            self.frames_ctrl.set_warning("Data is not doppler processed!")
        else:
            self.frames_ctrl.clear_state()


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