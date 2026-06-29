import numpy as np

import scipy

import matplotlib.pyplot as plt

N = 1000
fs = 50

t = np.arange(0,100)

x_tmp = np.mod(t,10)
x = np.zeros_like(t)
x[x_tmp>4] = 1
x[x_tmp<5] = 0


fs = 1

order = 4
# fc_DeTren_02 = 0.9
fc = 0.45
b, a = scipy.signal.butter(order, fc, btype='lowpass',fs = fs)
y = scipy.signal.lfilter(b, a, x)
y2 = scipy.signal.filtfilt(b, a, x)


plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax1.plot(t,x,label = "og signal")
ax1.plot(t,y,label = "filt")
ax1.plot(t,y2,label = "filt filt")
ax1.set_xlabel("t [s]")
ax1.set_ylabel("sigi ")
# ax1.set_title(f"Phase unwrapping for [{params["i_Frames_begin"]}:{params["i_Frames_end"]},{doppler_bin},{range_bin+params["i_Range_begin"]},{ele_bin},{azi_bin}] <- (i_Frames,i_Doppler,i_Range,i_Ele,i_Azi), DoA: {params["Channel_processing"]}, Doppler: {params["Doppler_processing"]} ")
ax1.legend()
ax1.grid(True)

plt.show()