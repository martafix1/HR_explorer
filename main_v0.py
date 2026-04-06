import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
)


import processing.DoA
from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process

import visuals.DoA_3Dmesh as DoA_3Dmesh
import visuals.PhaseUnWr_plot as PhaseUnWr_plot
import visuals.HR_STFT_plot as HR_STFT_plot
import visuals.DoA_2DAziPlane as DoA_2DAziPlane

params = { "i_Frames_begin":100,"i_Frames_end":600,
            "i_Range_begin":5,"i_Range_end":30,
            "i_Doppler_begin":0,"i_Doppler_end":32,
            "DoA_azi_N_elements":8,
            "DoA_ele_N_elements":2,
            "DoA_azi_range_degs":90, "DoA_ele_range_degs":30,
            "Channel_processing":"DoA_customFFT",
            # "Channel_processing":"None",
            "Doppler_processing":"None",

            "range_index2dist"  : 0.046,
            "frame_index2time"  : 5e-2,
            "doppler_index2freq"  : 1/(5.76* 1e-3), # Active chirping time 5.76 ms
            "doppler_index2vel"  : 0.157, # Velocity resolution, m/s
            "radarRotation_deg" : 0
                   }

filePath = "data/unR_meas_noHR_32_rdr227_humancenter_06-03-2026_14-30-14.npz" 
# filePath = "data/unR_meas_noHR_33_rdr227_humancenterlowsitting_06-03-2026_14-33-03.npz"

loadedData = loadNPZ(filePath)

frames = loadedData["frames"]

params = HR_process.defaultSliders(frames,params)

# params["i_Frames_end"] = 100


penteract, DoA_dict = HR_process.process_A(frames,params)

params["azi_points"] = DoA_dict["azi_mesh_range"]
params["ele_points"] = DoA_dict["ele_mesh_range"]


# 2D DoA
params_2D_DoA = dict(params)

params_2D_DoA["DoA_ele_N_elements"] = 1
params_2D_DoA["DoA_azi_N_elements"] = 8
params_2D_DoA["Doppler_processing"] = "FFT"
print("2D DoA processing: process_A")
penteract_2D_DoA, DoA_dict = HR_process.process_A(frames,params_2D_DoA)

params_2D_DoA["azi_points"] = DoA_dict["azi_mesh_range"]
params_2D_DoA["ele_points"] = DoA_dict["ele_mesh_range"]



app = QApplication(sys.argv)

# --- main window ---
main_window = QWidget()
main_window.setWindowTitle("Main App")
main_window.setFixedSize(250, 120)

layout = QVBoxLayout()

label = QLabel("Main application placeholder. Should load data, load data - chirp etc params and then launch plots")
layout.addWidget(label)



window_DoA_3Dmesh = DoA_3Dmesh.MeshPlotter3D(penteract[:,0,:,:,:],params)
window_DoA_3Dmesh.show()

widow_PhaseUnwr = PhaseUnWr_plot.PlotWindow()
widow_PhaseUnwr.update_newData(penteract[:,0,:,:,:],params)
widow_PhaseUnwr.show()

window_STFT = HR_STFT_plot.PlotWindow()
window_STFT.update_newData(penteract[:,0,:,:,:],params)
window_STFT.show()


window_DoA_2DAziPlane = DoA_2DAziPlane.PlotWindow()
window_DoA_2DAziPlane.update_newData(penteract_2D_DoA,params_2D_DoA)
window_DoA_2DAziPlane.show()


exit_btn = QPushButton("Exit Application")
exit_btn.clicked.connect(lambda : app.quit())
layout.addWidget(exit_btn)

main_window.setLayout(layout)
main_window.show()

# TODO make the heatmap STFT window, phase unwraping signgle signal window and link the sliders
# TODO linking the sliders can be done by putting the master sliders into main window

sys.exit(app.exec())
