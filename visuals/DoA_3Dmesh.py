from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from PySide6 import QtWidgets, QtCore

import pyqtgraph.opengl as gl
from matplotlib import colormaps
from pyqtgraph.Transform3D import Transform3D

from matplotlib import pyplot

import numpy as np
import visuals.utils as utils

"""
DOCS:

Data_4D: 4D ndarray in form data[N_Frames,N_range,N_ele,N_Azi] 

sliderLink: {"i_frame":0, "i_range":0}

highlightLink: {"i_azi":0, "i_ele":0}

Params: dict {  "range_index2dist"  : 6e-2,
                "frame_index2time"  : 5e-2,
                "azi_points" : np.ndarray...,
                "ele_points" : np.ndarray...,
                "i_Range_begin": 0, "i_Range_end": 67,
                "i_Frames_begin": 0, "i_Frames_end": 69,
                "radarRotation_deg": 0,
                } 

"""

def upsample_to_min_resolution(data, az_points, el_points, min_res=8):
    res_el, res_az = data.shape

    scale_el = max(1, int(np.ceil(min_res / res_el)))
    scale_az = max(1, int(np.ceil(min_res / res_az)))

    # Repeat data (nearest neighbor upsampling)
    data_up = np.repeat(np.repeat(data, scale_el, axis=0), scale_az, axis=1)

    # Track which cells are original
    mask = np.zeros_like(data_up, dtype=bool)
    for i in range(res_el):
        for j in range(res_az):
            mask[i*scale_el:(i+1)*scale_el,
                 j*scale_az:(j+1)*scale_az] = True

    # Interpolate new az/el points
    if az_points.shape[0] < 2:
        az_points = np.array([az_points[0],-az_points[0]]) 

    if el_points.shape[0]  < 2:
       el_points = np.array([el_points[0],-el_points[0]]) 
    

    az_up = np.linspace(az_points[0], az_points[-1], data_up.shape[1])
    el_up = np.linspace(el_points[0], el_points[-1], data_up.shape[0])

    return data_up, az_up, el_up, mask


def create_pixelated_mesh(data, az_points, el_points, radius, norm_MAX, norm_MIN):
    
    data, az_points, el_points, valid_mask = upsample_to_min_resolution(data, az_points, el_points,8)
    
    """Generates vertex, face, and color data for a spherical heatmap."""
    res_el,res_az = data.shape
    print(f"data shape  {data.shape} ")
    def get_edges(centers):
        if len(centers) > 1:
            diff = np.diff(centers) / 2
            return np.concatenate([[centers[0] - diff[0]], centers[:-1] + diff, [centers[-1] + diff[-1]]])
        return np.array([centers[0]-5, centers[0]+5])

    az_edges = np.radians(get_edges(az_points))
    el_edges = np.radians(get_edges(el_points))
    
    verts, faces, face_colors = [], [], []
    cmap = colormaps.get_cmap('viridis')
    norm_MAX = np.max([norm_MAX,np.max(data)])
    norm_MIN = np.min([norm_MIN,np.min(data)])
    print(f"global max, min : {norm_MAX},{norm_MIN}")
    norm_data = (data - norm_MIN) / (norm_MAX - norm_MIN + 1e-6)

    v_idx = 0
    for i in range(res_el):
        for j in range(res_az):
            az_l, az_h = az_edges[j], az_edges[j+1]
            el_l, el_h = el_edges[i], el_edges[i+1]
            
            # 4 Corners
            corners = [(el_l, az_l), (el_l, az_h), (el_h, az_h), (el_h, az_l)]
            for el, az in corners:
                x = radius * np.cos(el) * np.cos(az)
                y = radius * np.cos(el) * np.sin(az)
                z = radius * np.sin(el)
                verts.append([x, y, z])
            
            faces.append([v_idx, v_idx+1, v_idx+2])
            faces.append([v_idx, v_idx+2, v_idx+3])
            
            color = cmap(norm_data[i, j])

            # If this cell is NOT original → make it visually distinct
            if not valid_mask[i, j]:
                edge_color = (0, 0, 0, 1)  # black
                color = tuple(0.7 * np.array(color))  # dim it slightly
            else:
                edge_color = color
            
            face_colors.append(edge_color)
            face_colors.append(edge_color)
            v_idx += 4

    return np.array(verts), np.array(faces), np.array(face_colors)




class MeshPlotter3D(QWidget):
    def __init__(self,data4D,params):
        super().__init__()

        self.setWindowTitle("Plot Window")
        self.resize(1100, 800)

        main_layout = QVBoxLayout()

        controls_container = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(2, 2, 2, 2)  # tiny margins
        controls_layout.setSpacing(2)  # small vertical spacing between rows
        
        self._scrollWheel_filter = utils.SliderWheelFilter()

        # --- Row 1: Range slider ---
        range_container = QtWidgets.QWidget()
        range_layout = QtWidgets.QHBoxLayout(range_container)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(5)

        range_layout.addWidget(QtWidgets.QLabel("Adjust Range:"))

        self.slider_range = QtWidgets.QSlider(QtCore.Qt.Horizontal) # type: ignore
        self.slider_range.setRange(67,420)
        self.slider_range.setValue(69)
        self.slider_range.installEventFilter(self._scrollWheel_filter)
        self.slider_range.setFixedHeight(20)  # small slider height
        self.slider_range.valueChanged.connect(self.update_onSliderMove)
        range_layout.addWidget(self.slider_range)

        self.range_label_display = QtWidgets.QLabel("Value:")
        range_layout.addWidget(self.range_label_display)

        controls_layout.addWidget(range_container)

        # --- Row 2: Frame slider ---
        frame_container = QtWidgets.QWidget()
        frame_layout = QtWidgets.QHBoxLayout(frame_container)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(5)

        frame_layout.addWidget(QtWidgets.QLabel("Frame:"))

        self.slider_frame = QtWidgets.QSlider(QtCore.Qt.Horizontal) # type: ignore
        self.slider_frame.setRange(67,420)
        self.slider_frame.setValue(69)
        self.slider_frame.installEventFilter(self._scrollWheel_filter)
        self.slider_frame.setFixedHeight(20)
        self.slider_frame.valueChanged.connect(self.update_onSliderMove)
        frame_layout.addWidget(self.slider_frame)

        self.frame_label_display = QtWidgets.QLabel("0")
        frame_layout.addWidget(self.frame_label_display)

        controls_layout.addWidget(frame_container)

        # Force minimal vertical size on controls_container
        controls_container.setMaximumHeight(60)

        # Add controls container to main layout
        main_layout.addWidget(controls_container)


        # 2. 3D Plot
        view = gl.GLViewWidget()
        view.setBackgroundColor('w')
        main_layout.addWidget(view)

        # World Grid
        world_grid = gl.GLGridItem()
        world_grid.setColor((200, 200, 200, 255))
        view.addItem(world_grid)

        
        # Initialize Global Mesh (ONLY ONCE)
        self.heatmap_mesh = gl.GLMeshItem(
            smooth=False, 
            drawEdges=True, 
            edgeColor=(0, 0, 0, 1),
            computeNormals=False  # Keep this false to prevent the errors
        )
        # Placeholder
        self.sensor_pose = Transform3D()
        self.sensor_pose.translate(0, 0, 2) 
        self.sensor_pose.rotate(0, 1, 0, 0) #change rotation here

        self.heatmap_mesh.setTransform(self.sensor_pose)
        view.addItem(self.heatmap_mesh)

        # --- ADD SENSOR AXIS HERE ---
        self.sensor_axis = gl.GLAxisItem()
        self.sensor_axis.setSize(x=5, y=5, z=5) # Make it visible
        self.sensor_axis.setTransform(self.sensor_pose) # Match the mesh
        view.addItem(self.sensor_axis)
        # ----------------------------

        # Initialize with first draw

        self.setLayout(main_layout)

        self.update_newData(data4D,params)



    def update_onSliderMove(self):
        val_range = self.slider_range.value()
        val_frame = self.slider_frame.value()
        range_m = val_range*self.params["range_index2dist"]
        self.range_label_display.setText(f"idx: {val_range} ~ {range_m :.2f} m")
        
        
        frameTime_s = val_frame * self.params["frame_index2time"]
        self.frame_label_display.setText(f"idx: {val_frame} ~ {frameTime_s :.2f} s")
        # sliders = {"Frame": 10,"Range":17}
        
        new_data = self.data4D[val_frame,val_range,:,:]


        # range ampl compensation
        new_data = new_data * (range_m*range_m) # range^2 coz power is P = 1/range^4 but V ~ sqrt(P)

        norm_max = np.max(self.data4D[val_frame,:,:,:])
        norm_min = np.min(self.data4D[val_frame,:,:,:])
    
        
        v, f, c = create_pixelated_mesh(new_data, self.params["azi_points"],  self.params["ele_points"], radius=range_m, norm_MAX=norm_max,norm_MIN=norm_min)
        
        try:
            # Forcing a reset of the mesh data to prevent the 'NoneType' edge error
            self.heatmap_mesh.setMeshData(
                vertexes=v, 
                faces=f, 
                faceColors=c, 
                computeNormals=False
            )
        except:
            pass # Prevents the console flood during rapid GUI events
        

    def update_newData(self,data,params):
        self.data4D = np.abs(data)
        self.params = params

        self.slider_frame.setRange(self.params["i_Frames_begin"], self.params["i_Frames_end"]-1)
        self.slider_range.setRange(self.params["i_Range_begin"], self.params["i_Range_end"]-1)
        
        self.sensor_pose.rotate(self.params["radarRotation_deg"], 1, 0, 0)
        
        self.heatmap_mesh.setTransform(self.sensor_pose)
        self.sensor_axis.setTransform(self.sensor_pose) # Match the mesh

        self.update_onSliderMove()
         
        pass

    def linkSliders(self,sliderLink):
        self.slider_frame.setValue = sliderLink["i_frame"]
        self.slider_range.setValue = sliderLink["i_range"]    

