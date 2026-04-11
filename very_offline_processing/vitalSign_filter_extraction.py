
import numpy as np
import sys, os

import scipy.signal

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process

import matplotlib.pyplot as plt


params = {  "DoA_azi_N_elements":8,
            "DoA_ele_N_elements":1,
            "DoA_azi_range_degs":90, "DoA_ele_range_degs":30,
            # "Channel_processing":"DoA_customFFT",
            "Channel_processing":"None",
            "Doppler_processing":"FFT",

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




frames0 = 200
frames1 = -200

params["i_Frames_begin"] += frames0
params["i_Frames_end"] += frames1

params["i_Range_begin"] = 20
params["i_Range_end"] = 26



t0 = params["i_Frames_begin"] * params["frame_index2time"]
t1 = params["i_Frames_end"] * params["frame_index2time"]

t = np.linspace(t0,t1, params["i_Frames_end"] -params["i_Frames_begin"] )


doppler_bin = 0
ele_bin = 0
azi_bin = 3
range_bin = 25-  params["i_Range_begin"]


penteract, DoA_dict = HR_process.process_A(frames,params)

data_new = penteract[:,doppler_bin,range_bin,ele_bin,azi_bin]
phase = np.angle(data_new)
magnitude = np.abs(data_new)
phase_unwrp = np.unwrap(phase)


fs = 1/params["frame_index2time"]  # Hz sampling freq

fc_BR = (0.1,0.8)  # Hz, cutoff freq
order_BR = 4  # good default

b_BR, a_BR = scipy.signal.butter(order_BR, fc_BR, btype='bandpass',fs = fs)
BR_signal = scipy.signal.filtfilt(b_BR, a_BR, phase_unwrp)


fc_HR = (0.8,3)  # Hz, cutoff freq
order_HR = 4  # good default

b_HR, a_HR = scipy.signal.butter(order_HR, fc_HR, btype='bandpass',fs = fs)
HR_signal = scipy.signal.filtfilt(b_HR, a_HR, phase_unwrp)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t,BR_signal,label = "breath rate signal")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)
plt.title(f"Phase unwrapping for [{params["i_Frames_begin"]}:{params["i_Frames_end"]},{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- (i_Frames,i_Doppler,i_Range,i_Ele,i_Azi), DoA: {params["Channel_processing"]}, Doppler: {params["Doppler_processing"]} ")

plt.subplot(2, 1, 2)
plt.plot(t,HR_signal,label = "heart rate signal")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)


plt.tight_layout()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t,BR_signal,label = "breath rate signal")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)
plt.title(f"Phase unwrapping for [{params["i_Frames_begin"]}:{params["i_Frames_end"]},{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- (i_Frames,i_Doppler,i_Range,i_Ele,i_Azi), DoA: {params["Channel_processing"]}, Doppler: {params["Doppler_processing"]} ")


analytic_BR = scipy.signal.hilbert(BR_signal)
phase_analytic_BR = np.unwrap(np.angle(analytic_BR))
inst_freq_BR = np.diff(phase_analytic_BR) * fs / (2 * np.pi)

plt.subplot(2, 1, 2)
plt.plot(t[:-1],inst_freq_BR,label = "BR freq")
plt.xlabel("t [s]")
plt.ylabel("freq [Hz]")
plt.legend()
plt.grid(True)


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t,HR_signal,label = "heart rate signal")
plt.xlabel("t [s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.grid(True)
plt.title(f"Phase unwrapping for [{params["i_Frames_begin"]}:{params["i_Frames_end"]},{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- (i_Frames,i_Doppler,i_Range,i_Ele,i_Azi), DoA: {params["Channel_processing"]}, Doppler: {params["Doppler_processing"]} ")


analytic_HR = scipy.signal.hilbert(HR_signal)
phase_analytic_HR = np.unwrap(np.angle(analytic_HR))
inst_freq_HR = np.diff(phase_analytic_HR) * fs / (2 * np.pi)

plt.subplot(2, 1, 2)
plt.plot(t[:-1],inst_freq_HR,label = "HR freq")
plt.xlabel("t [s]")
plt.ylabel("freq [Hz]")
plt.legend()
plt.grid(True)


plt.show()