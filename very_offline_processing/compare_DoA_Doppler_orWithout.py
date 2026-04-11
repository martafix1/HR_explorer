
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process

import matplotlib.pyplot as plt



params = {  "DoA_azi_N_elements":8,
            "DoA_ele_N_elements":1,
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

testCase1_key = "Channel_processing"
testCase1_vals = ["DoA_customFFT","None"]

testCase2_key = "Doppler_processing"
testCase2_vals = ["FFT","None"]

frames0 = 200
frames1 = -200

params["i_Frames_begin"] += frames0
params["i_Frames_end"] += frames1

params["i_Range_begin"] = 24
params["i_Range_end"] = 27

plt.figure()

t0 = params["i_Frames_begin"] * params["frame_index2time"]
t1 = params["i_Frames_end"] * params["frame_index2time"]

x = np.linspace(t0,t1, params["i_Frames_end"] -params["i_Frames_begin"] )


doppler_bin = 0
ele_bin = 0
azi_bin = 3
range_bin = 24-  params["i_Range_begin"]


for tc1 in testCase1_vals:
    for tc2 in testCase2_vals:

        params[testCase1_key] = tc1
        params[testCase2_key] = tc2

        penteract, DoA_dict = HR_process.process_A(frames,params)

        # val_frame_begin* self.params["frame_index2time"]
        

        name = f"line: {tc1},{tc2}"
        data_new = penteract[:,doppler_bin,range_bin,ele_bin,azi_bin]
        phase = np.angle(data_new)
        magnitude = np.abs(data_new)
        y = np.unwrap(phase)
        plt.plot(x,y, label = name)




plt.title(f"Phase unwrapping for [:,{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- <-(i_Frames,i_Doppler,i_Range,i_Ele,i_Azi)")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")

# Legend & grid
plt.legend()
plt.grid(True)


plt.show()