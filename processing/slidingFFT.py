
import numpy as np




def _window(sample_len: int, window_sel: str) -> tuple[np.ndarray, bool]:
    """Return window and interpolation-log flag."""
    if window_sel == "hann":
        return np.hanning(sample_len), True
    if window_sel == "hamming":
        return np.hamming(sample_len), True
    if window_sel == "blackman":
        return np.blackman(sample_len), True
    if window_sel == "rect":
        return np.ones(sample_len), False
    print(f"Unknown window {window_sel} using rect instead")
    return np.ones(sample_len), False





def FFT_single( signal,frameIDX_end,
                fs=20.0, initFrames = 20, stepFrames = 10 , sig_sample_len = 80,
                fixedFFT_size = 200, freqRangeStart = 0.1, freqRangeStop = 2.0, parabolicInterpolation = True,
                window_sel = "rect" ):
    
    binStart = np.floor(freqRangeStart / (fs / fixedFFT_size))
    binEnd = np.ceil(freqRangeStop / (fs / fixedFFT_size)) 
    binStart = int(max(binStart,0)) 
    binEnd = int(min(binEnd,(fixedFFT_size/2) +1)) 

    if initFrames > frameIDX_end:
        return None
    if (frameIDX_end - initFrames)% stepFrames !=0:
        newframeIDX_end = frameIDX_end - ((frameIDX_end - initFrames)% stepFrames) 
        print(f"req. frameIDX_end not at correct step position {frameIDX_end}, rounding down to {newframeIDX_end}")
        frameIDX_end = newframeIDX_end

    backStop = max(0,frameIDX_end-sig_sample_len)

    
    sample = signal[backStop:frameIDX_end].astype(float, copy=True)
    sig_sample_len = len(sample)
    
    window, useLogInParabolic = _window(sig_sample_len, window_sel)
    sample_winded = sample * window # copy it so it does not change og signal
    coherent_gain = np.mean(window)

    padded = np.zeros(fixedFFT_size)
    padded[: min(sig_sample_len, fixedFFT_size)] = sample_winded[:fixedFFT_size]

    spect = np.fft.rfft(padded)
    # freqs = np.fft.rfftfreq(fixedFFT_size, d=1 / fs)
    amplitude = np.abs(spect) / sig_sample_len
    amplitude[1:-1] *= 2
    amplitude /= coherent_gain


    peak_idx = int(np.argmax(amplitude[binStart:binEnd]) + binStart)
    interp_offset = 0.0
    if parabolicInterpolation and 0 < peak_idx < len(amplitude) - 1:
        # Parabolic peak estimate around the selected FFT bin.
        if useLogInParabolic:
            smol = np.finfo(float).tiny
            alpha = np.log(max(amplitude[peak_idx - 1], smol))
            beta = np.log(max(amplitude[peak_idx], smol))
            gamma = np.log(max(amplitude[peak_idx + 1], smol))
        else:
            alpha = amplitude[peak_idx - 1]
            beta = amplitude[peak_idx]
            gamma = amplitude[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if denom != 0:
            interp_offset = float(np.clip(0.5 * (alpha - gamma) / denom, -0.5, 0.5))
        peak_freq = (peak_idx + interp_offset) * fs / fixedFFT_size
        peak_amp = beta - 0.25 * (alpha - gamma) * interp_offset
        if useLogInParabolic:
            peak_amp = np.exp(peak_amp)

            
    else:
        peak_freq = peak_idx * (fs / fixedFFT_size)
        peak_amp = amplitude[peak_idx]

    return peak_freq, peak_amp, amplitude




def instantaneous_analysis_FFT(signal, fs=20.0, initFrames = 20, stepFrames = 10 , sig_sample = 80,
                                fixedFFT_size = 200, freqRangeStart = 0.1, freqRangeStop = 2.0, parabolicInterpolation = True,
                                window_sel = "rect" ):


    instFreqs = np.zeros_like(signal) 
    instAmpls = np.zeros_like(signal)
    lastIdx = 0 # to fill the arrays correctly

 
    for i in range(initFrames,len(signal),stepFrames):
        

        answer = FFT_single(signal,i,fs,initFrames,stepFrames,sig_sample,fixedFFT_size,freqRangeStart,freqRangeStop,parabolicInterpolation,window_sel)
        if answer is None:
            continue
        else:
            peak_freq, peak_amp, discard = answer
         

        instFreqs[lastIdx:i] = peak_freq
        instAmpls[lastIdx:i] = peak_amp

        lastIdx = i

    return instFreqs,instAmpls


