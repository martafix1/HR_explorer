import numpy as np
import os, sys,types
from typing import Callable, Any
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt

import very_offline_processing.generate_signal as sigGen


fs = 20
t, sig, BR_instantf, HR_instantf = sigGen.generate_composite_signal(fs=fs)

step_frames = 10
initStep_frames = 20
maxWinLen = 80 

fixedfft_size = 400 # 20s
freqs = np.fft.rfftfreq(fixedfft_size, d=1/fs)

instFreqs = np.zeros_like(sig) 
instAmpls = np.zeros_like(sig)
lastIdx = 0 # to fill the arrays correctly
for i in range(initStep_frames,len(sig),step_frames):

    
    backStop = max(0,i-maxWinLen)
    sampleLen = i-backStop

    sample = sig[backStop:i]
    
    padded = np.zeros(fixedfft_size)
    padded[:sampleLen] = sample

    spect = np.fft.rfft(padded)
    amplitude = np.abs(spect) /sampleLen # norm to sample size
    amplitude *=2 # half spectrum

    idx = np.argmax(amplitude)
    
    # Rife interpolation
    if amplitude[idx+1] > amplitude[idx-1]:
        p = 1
    else:
        p = -1

    delta = amplitude[idx+p] / (amplitude[idx] + amplitude[idx+p])
    
    # Interpolated bin
    k_interp = idx + p * delta

    peak_freq = freqs[idx]
    peak_amp = amplitude[idx]

    instFreqs[lastIdx:i] = k_interp * fs / fixedfft_size
    instAmpls[lastIdx:i] = peak_amp

    lastIdx = i

    # plt.figure(figsize=(8, 4))
    # plt.plot(freqs, amplitude)
    # plt.title(f"FFT using first {backStop}:{i}, (zero padded to {fixedfft_size})")
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.xlim(0, fs/2)
    # plt.tight_layout()



plt.figure(figsize=(8, 4))
plt.plot(t, sig, label="signal")
plt.plot(t, instFreqs, label="freq")
plt.plot(t, BR_instantf, label="freq ref", linestyle='--')
plt.plot(t, instAmpls, label="ampl")
plt.grid(True)
plt.legend()

plt.show()






