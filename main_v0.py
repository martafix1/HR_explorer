import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
)


import processing.DoA
from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process

import visuals.DoA_3Dmesh as DoA_3Dmesh
# import visuals.PhaseUnWr_plot as PhaseUnWr_plot
import visuals.PhaseUnWr_plot_2 as PhaseUnWr_plot
# import visuals.HR_STFT_plot as HR_STFT_plot
import visuals.HR_STFT_plot_2 as HR_STFT_plot
import visuals.DoA_2DAziPlane as DoA_2DAziPlane
# import visuals.DoA_2DAziPlane_tracking as DoA_2DTracking
import visuals.DoA_2DAziPlane_tracking as DoA_2DTracking

import visuals.TrackedSignals_plot as TrackingPlot

import visuals.param_controls as pctrl

params = { "i_Frames_begin":100,"i_Frames_end":600,
            "i_Range_begin":5,"i_Range_end":30,
            "i_Doppler_begin":0,"i_Doppler_end":32,
            "DoA_azi_N_elements":8,
            "DoA_ele_N_elements":1,
            "DoA_azi_range_degs":90, "DoA_ele_range_degs":30,
            "Channel_processing":"DoA_customFFT",
            # "Channel_processing":"None",
            "Doppler_processing":"FFT",

            "range_index2dist"  : 0.046,
            "frame_index2time"  : 5e-2,
            "doppler_index2freq"  : 1/(5.76* 1e-3), # Active chirping time 5.76 ms
            "doppler_index2vel"  : 0.157, # Velocity resolution, m/s
            "radarRotation_deg" : 0
                   }

filePath = "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_16_michalautoklid_24-04-2026_13-41-17.npz"

# filePath = "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_18_michalauto_jizda_24-04-2026_14-00-40.npz"

filePath = "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_21_michalauto_radarvodorovne_stani_24-04-2026_14-32-50.npz"
# filePath = "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_22_michalauto_radarvodorovne_jizda_rucedolekdyztoslo_24-04-2026_14-39-26.npz"
# 15 unR_meas_15_michalautoklid_24-04-2026_13-40-12.npz
# 16 unR_meas_16_michalautoklid_24-04-2026_13-41-17.npz
# 17 unR_meas_17_michalauto_motorbezi_24-04-2026_13-47-15.npz
# 18 unR_meas_18_michalauto_jizda_24-04-2026_14-00-40.npz
# 19 unR_meas_19_michalauto_jizda_rucehore_24-04-2026_14-08-41.npz
# 20 unR_meas_20_michalauto_jizda_rucehore_normalnipohybynavic_24-04-2026_14-15-53.npz  
# 21 unR_meas_21_michalauto_radarvodorovne_stani_24-04-2026_14-32-50.npz
# 22 unR_meas_22_michalauto_radarvodorovne_jizda_rucedolekdyztoslo_24-04-2026_14-39-26.npz

# filePath = "data/unR_meas_noHR_32_rdr227_humancenter_06-03-2026_14-30-14.npz" 
# filePath = "data/unR_meas_noHR_33_rdr227_humancenterlowsitting_06-03-2026_14-33-03.npz"

# filePath = "data/unR_meas_7_josef_motorbezi_08-08-2025_13-38-58.npz"
# filePath = "data/unR_meas_9_josef_jizda_08-08-2025_13-43-06.npz"

loadedData = loadNPZ(filePath)

frames = loadedData["frames"]

params = HR_process.defaultSliders(frames,params)

# params["i_Frames_begin"] = 1000
# params["i_Frames_end"] = 1500


# penteract, DoA_dict = HR_process.process_A(frames,params)

# params["azi_points"] = DoA_dict["azi_mesh_range"]
# params["ele_points"] = DoA_dict["ele_mesh_range"]


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



# window_DoA_3Dmesh = DoA_3Dmesh.MeshPlotter3D(penteract[:,0,:,:,:],params)
# window_DoA_3Dmesh.show()

widow_PhaseUnwr = PhaseUnWr_plot.PlotWindow()
widow_PhaseUnwr.update_newData(penteract_2D_DoA[:,:,:,:,:],params_2D_DoA)
widow_PhaseUnwr.show()

# window_STFT = HR_STFT_plot.PlotWindow()
# window_STFT.update_newData(penteract_2D_DoA[:,0,:,:,:],params_2D_DoA)
# window_STFT.show()


# window_DoA_2DAziPlane = DoA_2DAziPlane.PlotWindow()
# window_DoA_2DAziPlane.update_newData(penteract_2D_DoA,params_2D_DoA)
# window_DoA_2DAziPlane.show()

window_tracking =  DoA_2DTracking.PlotWindow()
window_tracking.update_newData(penteract_2D_DoA,params_2D_DoA)
window_tracking.show()

window_trackPlotting = TrackingPlot.PlotWindow()
window_trackPlotting.update_newParams(params_2D_DoA)
window_trackPlotting.assignDataRetrievingFunction(window_tracking.returnTrackedSignals)
window_trackPlotting.show()


exit_btn = QPushButton("Exit Application")
exit_btn.clicked.connect(lambda : app.quit())
layout.addWidget(exit_btn)

main_window.setLayout(layout)
main_window.show()

# TODO make the heatmap STFT window, phase unwraping signgle signal window and link the sliders
# TODO linking the sliders can be done by putting the master sliders into main window

sys.exit(app.exec())
