
import numpy as np



def instantaneous_analysis_FFT(signal, fs=20, initFrames = 20, stepFrames = 10 , sig_sample = 80,
                                fixedFFT_size = 200, freqRangeStart = 0.1, freqRangeStop = 2, parabolicInterpolation = True,
                                window_sel = "rect" ):


    instFreqs = np.zeros_like(signal) 
    instAmpls = np.zeros_like(signal)
    lastIdx = 0 # to fill the arrays correctly

    # freqs = np.fft.rfftfreq(fixedFFT_size, d=1/fs)
    binStart = np.floor(freqRangeStart / (fs / fixedFFT_size))
    binEnd = np.ceil(freqRangeStop / (fs / fixedFFT_size)) 
    binStart = int(max(binStart,0)) 
    binEnd = int(min(binEnd,(fixedFFT_size/2) +1)) 
    for i in range(initFrames,len(signal),stepFrames):

        useLogInParabolic = False
        backStop = max(0,i-sig_sample)
        sampleLen = i-backStop
        if window_sel == "hann":
            window = np.hanning(sampleLen)
            useLogInParabolic = True
        elif window_sel == "hamming":
            window = np.hamming(sampleLen)
            useLogInParabolic = True
        elif window_sel == "rect":
            window = np.ones(sampleLen)
        else:
            print(f"Unknown window {window_sel} using rect instead")
            window = np.ones(sampleLen)

        sample = signal[backStop:i]
        
        sample *= window

        padded = np.zeros(fixedFFT_size)
        padded[:sampleLen] = sample

        spect = np.fft.rfft(padded)
        amplitude = np.abs(spect) /sampleLen # norm to sample size
        amplitude *=2 # half of spectrum

        idx = np.argmax(amplitude[binStart:binEnd] ) + binStart

        if parabolicInterpolation and 0 < idx < len(amplitude)-1:
            # Parabolic interpolation
            if useLogInParabolic:
                alpha = np.log(amplitude[idx-1]) 
                beta  = np.log(amplitude[idx] )
                gamma = np.log(amplitude[idx+1])
            else:
                alpha = amplitude[idx-1]
                beta  = amplitude[idx]
                gamma = amplitude[idx+1]
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)

            p = np.clip(p, -0.5, 0.5) # in case of badly defined peak saturate it
            f_est = (idx + p) * fs / fixedFFT_size
            A_est = beta - 0.25 * (alpha - gamma) * p
            
            peak_freq = f_est
            peak_amp = A_est

            # Rife interpolation
            # if amplitude[idx+1] > amplitude[idx-1]:
            #     p = 1
            # else:
            #     p = -1
            # delta = amplitude[idx+p] / (amplitude[idx] + amplitude[idx+p])
            # k_interp = idx + p * delta  # Interpolated bin
        else:
            peak_freq = idx * (fs / fixedFFT_size)
            peak_amp = amplitude[idx]

        instFreqs[lastIdx:i] = peak_freq
        instAmpls[lastIdx:i] = peak_amp

        lastIdx = i

    return instFreqs,instAmpls


