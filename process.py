import os, sys
from pathlib import Path
import numpy as np
import time

# import DoA_testDataGen

# import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph.opengl as gl
from matplotlib import colormaps
from pyqtgraph.Transform3D import Transform3D

from matplotlib import pyplot as plt

print("---- module import context ----")
print("module __file__:", Path(__file__).resolve())
print("cwd:", Path.cwd())
print("sys.path[0]:", sys.path[0])
print("first 10 sys.path entries:")
for p in sys.path[:10]:
    print("   ", p)
print("--------------------------------")

from loadNPZ import loadNPZ

# from DoA import create_DFT_matrix
import DoA
import HR_process
import HR_window

# filePath = "Projects/TestProject/analysis_modules/plot_radar_ampl/data/meas7_unrosed.npz"
#front:
# filePath = "Projects/TestProject/analysis_modules/plot_radar_ampl/data/unR_meas_noHR_29_rdr227_directfront_06-03-2026_14-20-25.npz"
#left:
# filePath = "Projects/TestProject/analysis_modules/plot_radar_ampl/data/unR_meas_noHR_30_rdr227_left_06-03-2026_14-23-22.npz"
#right:
# filePath = "Projects/TestProject/analysis_modules/plot_radar_ampl/data/unR_meas_noHR_31_rdr227_right_06-03-2026_14-26-33.npz"
#human centre:
filePath = "Projects/TestProject/analysis_modules/plot_radar_ampl/data/unR_meas_noHR_32_rdr227_humancenter_06-03-2026_14-30-14.npz"
reqFrame_glob = 300

def loadData():
    # filePath = "Projects/TestProject/analysis_modules/plot_radar_ampl/data/meas7_unrosed.npz"
    data = loadNPZ(filePath)
    return data

def setup_process():
    pass

# --- Global Config / Data ---
TEST_AZ = np.linspace(-90, 90, 8)
TEST_EL = np.linspace(-15, 15, 2)
loadedData = None
doa_dict : np.ndarray = np.empty(1)
globalDataNorm_max  = 1.0
globalDataNorm_min  = 1e9

sensor_pose = Transform3D()
sensor_pose.translate(0, 0, 2) 
sensor_pose.rotate(0, 1, 0, 0) #change rotation here

penteract = np.zeros((1,1,1,1,1))

def process(data, sliders):
    """Process input data with slider parameters.
    
    Args:
        data: Input dataset (e.g., loaded from file)
        sliders: dict of slider values {name: value, ...}
    
    Returns:
        dict with processed data keys (e.g., {"output": ...})
    """

    frames : np.ndarray = data["frames"] 
    req_frame = sliders["Frame"]
    # req_chirp = sliders["Chirp"] # not gonna go trouh all velocities searching for a target
    req_range = sliders["Range"]
    #frames[req_frame,req_chirp]
    
    ## data = data.reshape([params['n_chirps'], params['n_channels'], params['n_samples']])
    ## frames.append(data)
    
    # # full on processing is slow, but is done this way (bellow). Instead use only part of interest.
    # print(f"frames size: {frames.shape}")
    # frames_range = np.fft.fft(frames,axis=3) # range fft
    # frames_range_doppler = np.fft.fft(frames_range,axis=1) # velocity fft
    # frames_range_SumDopp = np.sum(frames_range_doppler,axis=1) # agregate velocities by sum
    # frames_ToBeSphereFront_line = frames_range_SumDopp[req_frame,:,req_range]
    # print(f"frames_range_doppler size: {frames_range_doppler.shape}")
    # print(f"frames_range_SumDopp size: {frames_range_SumDopp.shape}")
    # print(f"sphereFront_line size: {frames_ToBeSphereFront_line.shape}")
    
    # no giant FFT needed:
    pStart = time.perf_counter()
    rawData_lockedFrame = frames[req_frame,:,:,:]
    rangeData_lockedFrame = np.fft.fft(rawData_lockedFrame,axis=2) # range fft
    pRfftDone = time.perf_counter()

    # print("TESTING DATA")
    # rangeData_lockedFrame = DoA_testDataGen.Cube_rangeFFT*1e6
    rangeData_lockedFrameRange = rangeData_lockedFrame[:,:,req_range]
    range_dopplerData_lockedFrameRange = np.fft.fft(rangeData_lockedFrameRange,axis=0) # velocity fft
    #print(f"doppler freq: {np.fft.fftfreq(rangeData_lockedFrameRange.shape[0])} ")
    #range_dopplerData_lockedFrameRange_aggreageVelocity = np.sum(range_dopplerData_lockedFrameRange,axis=0) # agregate velocities by sum
    range_dopplerData_lockedFrameRange_aggreageVelocity = range_dopplerData_lockedFrameRange[0,:] # select zero velocity
    pDone = time.perf_counter()
    sphereFront_wannaBe_Data_line = range_dopplerData_lockedFrameRange_aggreageVelocity


    #doa_dict = create_DFT_matrix(DoA.params)
    doa_dict = DoA.DoA_dict_precalc

    # print("TESTING DATA")
    # # this dont work, WHY?
    # # steering_spectrum = np.abs(np.matmul(sphereFront_wannaBe_Data_line/1, doa_dict["beam_vector_flat"] ).reshape([len(doa_dict["azi_mesh_range"]), len(doa_dict["ele_mesh_range"])]))
    # # this works, WHY?
    # steering_spectrum = np.abs(np.matmul(sphereFront_wannaBe_Data_line/1, doa_dict["beam_vector_flat"] ).reshape( [len(doa_dict["ele_mesh_range"]),len(doa_dict["azi_mesh_range"])]))
    # # plt.imshow(steering_spectrum)
    # # plt.show()
    

    steering_spectrum = np.abs(np.matmul(sphereFront_wannaBe_Data_line/doa_dict["calib"], doa_dict["beam_vector_flat"] ).reshape([len(doa_dict["ele_mesh_range"]), len(doa_dict["azi_mesh_range"])]))
    pSpectDone = time.perf_counter()

    # print(f"Total processing time [ms]: {(pSpectDone - pStart)*1000} ; rangeFFT: {(pRfftDone-pStart)*1000} dopplerFFT: {(pDone-pRfftDone)*1000} DOA: {(pSpectDone-pDone)*1000}  ")
    

    return {"output": steering_spectrum} # return {"output": frames[req_frame]}

# just for offline testing
def plot(data,sliders,view):

    pass




def create_pixelated_mesh(data, az_points, el_points, radius):
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
    global globalDataNorm_max, globalDataNorm_min
    globalDataNorm_max = np.max([globalDataNorm_max,np.max(data)])
    globalDataNorm_min = np.min([globalDataNorm_min,np.min(data)])
    print(f"global max, min : {globalDataNorm_max},{globalDataNorm_min}")
    norm_data = (data - globalDataNorm_min) / (globalDataNorm_max - globalDataNorm_min + 1e-6)

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
            face_colors.append(color)
            face_colors.append(color)
            v_idx += 4

    return np.array(verts), np.array(faces), np.array(face_colors)

def update_handler():
    global loadData, penteract, doa_dict
    val_range = slider_range.value()
    val_frame = slider_frame.value()
    rangeMultiplier = 0.06
    range_m = val_range*rangeMultiplier
    range_label_display.setText(f"idx: {val_range} ~ {range_m :.2f} m")
    
    framerateMultiplier = 0.05
    frameTime_s = val_frame * framerateMultiplier
    frame_label_display.setText(f"idx: {val_frame} ~ {frameTime_s :.2f} s")
    sliders = {"Frame": val_frame,"Range":val_range}
    # sliders = {"Frame": 10,"Range":17}
    
    new_data = penteract[val_frame,0,val_range,:,:]


    #out = process(data=loadedData,sliders= sliders)
    # new_data = out["output"]

    # range ampl compensation
    new_data = new_data * (range_m*range_m) # range^2 coz power is P = 1/range^4 but V ~ sqrt(P)

    #new_data = np.random.random((2, 8))
    # print("TEST PLOTTING")
    # new_data = np.zeros([doa_dict["azi_mesh_range"].shape[-1],doa_dict["ele_mesh_range"].shape[-1]])
    # lastval = 0
    # for azi in range(doa_dict["azi_mesh_range"].shape[-1]):
    #     for ele in range(doa_dict["ele_mesh_range"].shape[-1]):
    #         new_data[azi,ele] = lastval
    #         lastval +=1
    #     pass
    # print(new_data)
    
    
    v, f, c = create_pixelated_mesh(new_data, doa_dict["azi_mesh_range"],  doa_dict["ele_mesh_range"], radius=range_m)
    
    try:
        # Forcing a reset of the mesh data to prevent the 'NoneType' edge error
        heatmap_mesh.setMeshData(
            vertexes=v, 
            faces=f, 
            faceColors=c, 
            computeNormals=False
        )
    except:
        pass # Prevents the console flood during rapid GUI events


if __name__ == "__main__":
    
    loadedData = loadData()
    frames = loadedData["frames"]
    HR_sliders = HR_process.sliders
    HR_sliders = HR_process.defaultSliders(frames,HR_sliders)
    HR_sliders["i_Frames_end"] = 300
    penteract, doa_dict = HR_process.process_A(frames,HR_sliders)
    
    app = QtWidgets.QApplication(sys.argv)

    window = QtWidgets.QMainWindow()
    window.resize(1100, 800)
    central_widget = QtWidgets.QWidget()
    window.setCentralWidget(central_widget)
    main_layout = QtWidgets.QVBoxLayout(central_widget)

    controls_container = QtWidgets.QWidget()
    controls_layout = QtWidgets.QVBoxLayout(controls_container)
    controls_layout.setContentsMargins(2, 2, 2, 2)  # tiny margins
    controls_layout.setSpacing(2)  # small vertical spacing between rows
    

    # --- Row 1: Range slider ---
    range_container = QtWidgets.QWidget()
    range_layout = QtWidgets.QHBoxLayout(range_container)
    range_layout.setContentsMargins(0, 0, 0, 0)
    range_layout.setSpacing(5)

    range_layout.addWidget(QtWidgets.QLabel("Adjust Range:"))

    slider_range = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_range.setRange(HR_sliders["i_Range_begin"], HR_sliders["i_Range_end"]-1)
    slider_range.setValue(HR_sliders["i_Range_begin"])
    slider_range.setFixedHeight(20)  # small slider height
    slider_range.valueChanged.connect(update_handler)
    range_layout.addWidget(slider_range)

    range_label_display = QtWidgets.QLabel("Value:")
    range_layout.addWidget(range_label_display)

    range_container.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding,
        QtWidgets.QSizePolicy.Minimum
    )

    controls_layout.addWidget(range_container)

    # --- Row 2: Frame slider ---
    frame_container = QtWidgets.QWidget()
    frame_layout = QtWidgets.QHBoxLayout(frame_container)
    frame_layout.setContentsMargins(0, 0, 0, 0)
    frame_layout.setSpacing(5)

    frame_layout.addWidget(QtWidgets.QLabel("Frame:"))

    slider_frame = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_frame.setRange(HR_sliders["i_Frames_begin"], HR_sliders["i_Frames_end"]-1)
    slider_frame.setValue(HR_sliders["i_Frames_begin"])
    slider_frame.setFixedHeight(20)
    slider_frame.valueChanged.connect(update_handler)
    frame_layout.addWidget(slider_frame)

    frame_label_display = QtWidgets.QLabel("0")
    frame_layout.addWidget(frame_label_display)

    frame_container.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding,
        QtWidgets.QSizePolicy.Minimum
    )

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

    # Sensor Pose Definition
    # sensor_pose = Transform3D() # done as global
    # sensor_pose.translate(0, 0, 2) 
    # sensor_pose.rotate(90, 1, 0, 0)


    # Initialize Global Mesh (ONLY ONCE)
    heatmap_mesh = gl.GLMeshItem(
        smooth=False, 
        drawEdges=True, 
        edgeColor=(0, 0, 0, 1),
        computeNormals=False  # Keep this false to prevent the errors
    )
    heatmap_mesh.setTransform(sensor_pose)
    view.addItem(heatmap_mesh)

    # --- ADD SENSOR AXIS HERE ---
    sensor_axis = gl.GLAxisItem()
    sensor_axis.setSize(x=5, y=5, z=5) # Make it visible
    sensor_axis.setTransform(sensor_pose) # Match the mesh
    view.addItem(sensor_axis)
    # ----------------------------

    # Initialize with first draw
    update_handler()

    window.show()
    sys.exit(app.exec())






# if __name__ == "__main__":
#     sliders = {"Frame": 10,"Range":10}
#     # loadStart = time.perf_counter()
#     # data = loadData()
#     # loadDone = time.perf_counter()
#     # print(f"Loading time [s]: {loadDone-loadStart :.3f}")
#     # out = process(data,sliders)
#     # print(out)

#     ## plot and app stuff
#     out = None
#     app = QtWidgets.QApplication(sys.argv)
#     view = gl.GLViewWidget()
#     plot(out,sliders,view)
#     sys.exit(app.exec())
#     pass