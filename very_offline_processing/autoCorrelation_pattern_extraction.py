import numpy as np
import scipy
import matplotlib.pyplot as plt
rng = np.random.default_rng()



import numpy as np
import matplotlib.pyplot as plt


def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray))

# Example pattern functions (you can replace these!)
def ekg_like(x):
    """Simple EKG-like spike"""
    return np.exp(-50 * (x - 0.2)**2) - 0.5*np.exp(-80 * (x - 0.25)**2)

def trapezoid_smooth(phase,flatDuty = 0.5, initPhase=0.0, lowPass_freq = 0.1):
    single_rampDuty = (1-flatDuty)/2.0
    single_flatDuty = (flatDuty)/2.0
    r1_start = (0 ) 
    r1_end =  (  single_rampDuty ) 
    r2_start = (  single_flatDuty + single_rampDuty ) 
    r2_end = ( single_flatDuty + single_rampDuty*2 ) 
    if is_array_like(phase):
        output = np.empty_like(phase)
    else:
        output = np.zeros(1)
    for i,fi in enumerate(phase):
        fi = (fi-initPhase) % 1

        if(r1_start< fi and fi  < r1_end):
            output[i] = (fi-r1_start)/(r1_end - r1_start)
        elif(r1_end <= fi and fi  <= r2_start):
            output[i] = 1
        elif(r2_start< fi and fi  < r2_end):
            output[i] = (r2_end-fi)/(r2_end - r2_start)
        else:
            output[i] = 0
    
    order = 2
    fc = lowPass_freq
    b, a = scipy.signal.butter(order, fc, btype='lowpass',fs = fs)
    output = scipy.signal.filtfilt(b, a, output)
    return output
    
# input: t in seconds, array of times; fs just so i dont have to calculate it again
def breath_sig_trapezoidFreqVary(t,fs):
    # output = np.empty_like(t)
    
   
    instantaneous_f_br = 0.3 + (trapezoid_smooth(t/20,initPhase=0.75,lowPass_freq=0.2) * 0.15)-0.05
    # constant one
    # instantaneous_f_br = 0.3 + 0*t # needs the t to make it an array

    # Phase is the integral of instantaneous frequency
    phase_br_radians = 2 * np.pi * np.cumsum(instantaneous_f_br) / fs 

    harmonics_signal = ( #values read from Breath spectrum
        3.66 * 1 * np.sin(1 * phase_br_radians + 2.56) +
        0.91 * 1 * np.sin(2 * phase_br_radians + 2.17) +
        0.22 * 1 * np.sin(3 * phase_br_radians + -0.51) +
        0.23 * 1 * np.sin(4 * phase_br_radians + -1.1)
    ) / (3.66)

    return harmonics_signal, instantaneous_f_br

def heart_sig_trapezoidFreqVary(t,fs):
    

    # instantPhase = t*1 # constant freq

    instantaneous_f_hr = 1 + (trapezoid_smooth(t/40,initPhase=0.75) * 0.2)-0.05

    pulse_duration = 0.4
    viablePulseDuration = 0.5 # it makes a pulse and then it goes to zero forever after roughly 0.5-0.6s 

    instantaneous_phase_cycles =  np.cumsum(instantaneous_f_hr) / fs 

    instantaneous_phase_cycles = instantaneous_phase_cycles % 1

    hr_sig = ekg_like(instantaneous_phase_cycles)/0.65 # to normilize it to 1

    return hr_sig, instantaneous_f_hr



# Time axis
fs = 20  # sampling frequency
T = 60      # total duration (seconds)
t = np.linspace(0, T, int(fs*T), endpoint=False)


BreathSig,BR_instantf = breath_sig_trapezoidFreqVary(t,fs)

HeartSig, HR_instantf = heart_sig_trapezoidFreqVary(t,fs)

sig = (HeartSig*0.1) + BreathSig



# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, sig, label="Signal", linewidth=1)
# plt.plot(t, BreathSig, label="Signal", linewidth=1)
plt.plot(t, BR_instantf, '--', label="inst f BR")
# plt.plot(t, pattern2, '--', label="Pattern 2")
# plt.plot(t, HeartSig, label="HeartSig", linewidth=1)
plt.plot(t, HR_instantf, '--', label="inst f HR")

plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Title")
plt.tight_layout()
plt.grid(True)
# plt.show()


# sliding window autocorrelation
window_begin = 20
window_end = 100
win_size = window_end-window_begin
print(f"win size: {win_size/fs} [s]")
run = True
i = 0
# while(run):

i+=1
# result = scipy.signal.correlate(sig, sig, mode='same')
result = scipy.signal.correlate(sig, sig[window_begin:window_end], mode='valid')

# result2 = result*result # we actually dont want to increase the frequency two fold i think

peaks, properties = scipy.signal.find_peaks(result, height=2, distance=2)

peaks_inSeconds = (peaks + win_size/2)/fs
print(peaks_inSeconds)
print(properties)

## correlation spectra
x = result
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), d=1/fs)

scale = 1 #np.sum(w) / len(x) # correction for window (which i did not use (= rectangle))
amplitude = np.abs(X) / (len(x) * scale)
amplitude[1:-1] *= 2  # single-sided spectrum correction

phase = np.angle(X)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(10, 6))
fig.tight_layout()

line1, = ax1.plot(t[int((win_size-1)/2):-int((win_size/2))],result)
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Time [s]")
ax1.set_title("CrossCorrelation")
ax1.grid(True)
    
line2, = ax2.plot(freqs, amplitude)
ax2.set_ylabel("Amplitude")
ax2.set_title("Amplitude Spectrum")
ax2.grid(True)

line3, = ax3.plot(freqs, phase)
ax3.set_ylabel("Phase [rad]")
ax3.set_xlabel("Frequency [Hz]")
ax3.set_title("Phase Spectrum")
ax3.grid(True)
ax3.sharex(ax2)

## ===================== custom "skew model" for pattern fitting.


from scipy.optimize import least_squares
from scipy.interpolate import interp1d

# f = interp1d(t_signal, x_signal, kind='cubic', fill_value="extrapolate")

def model_stretch(params, t_ref, curve):
    alpha, tau  = params
    t_trans = alpha * t_ref + tau
    x_warped = curve(t_trans)
    return x_warped

def model_stretch_skew(params, t_ref,curve):
    alpha, tau, a0, a1, b = params

    # time transform
    t_trans = alpha * t_ref + tau
    # evaluate signal
    x_warped = curve(t_trans)
    # normalized time (0 → 1)
    t_norm = (t_ref - t_ref.min()) / (t_ref.max() - t_ref.min())
    # amplitude scaling (linear ramp)
    a_t = a0 + (a1 - a0) * t_norm
    return a_t * x_warped + b

def residuals_stretch(params, t_ref, y_ref,curve):
    return model_stretch(params, t_ref,curve) - y_ref

def residuals_stretch_skew(params, t_ref, y_ref,curve):
    return model_stretch_skew(params, t_ref,curve) - y_ref

# res = least_squares(residuals, x0, args=(t_ref, y_ref,curve))
# params_opt = res.x

## ==== figgure it out, it needs to happen in the loop.

# plot correlation for a peak
peak_of_interest_idx = 1

fig, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
line1, = ax1.plot(t,sig )
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Time [s]")
# ax1.set_title(f"Peak aligment correlation, peak @ {peaks_inSeconds[peak_of_interest_idx]} s")
ax1.set_title(f"Peak aligment correlation, many peaks")
ax1.grid(True)

selected_pattern = sig[window_begin:window_end]
curve = scipy.interpolate.interp1d(np.arange(0,win_size)/fs, selected_pattern, kind='cubic',bounds_error=False,fill_value=0.0
)

pattern_sum = np.zeros(win_size)



for idx,ignoring in enumerate(peaks):
    peak_of_interest_idx = idx
    sample_time_range = np.arange(peaks[peak_of_interest_idx],peaks[peak_of_interest_idx]+win_size) /fs # range(peaks[0],peaks[0]+win_size) /fs
    line2, = ax2.plot(sample_time_range,sig[window_begin:window_end] )

    rangeExpand = 20 # 1s range expand
    local_pattern_toFit = sig[peaks[peak_of_interest_idx]-rangeExpand:peaks[peak_of_interest_idx]+win_size+rangeExpand]
    y_ref = local_pattern_toFit
    t_ref = np.arange(len(y_ref)) / fs
    x0 = [1.0, 0.0] 
    bounds = ([0.3,- win_size/fs ],
              [3,    win_size/fs ])
    res = least_squares(residuals_stretch, x0,bounds=bounds, args=(t_ref, y_ref,curve))
    alfa,tau = res.x
    print(f"i: {idx} - a: {alfa}, tau {tau} [s]")
    new_t = np.arange(0+tau)

    sig_aligned = curve(alfa * t + tau)
    ax1.plot(t + (peaks[peak_of_interest_idx]-rangeExpand)/fs  ,sig_aligned)

    local_pattern = sig[peaks[peak_of_interest_idx]:peaks[peak_of_interest_idx]+win_size]

    ax3.plot(sample_time_range,local_pattern)
    pattern_sum += local_pattern
    pass
ax2.set_ylabel("Amplitude")
ax2.set_xlabel("Time [s]")
ax2.grid(True)

ax3.set_ylabel("Amplitude")
ax3.set_xlabel("Time [s]")
ax3.grid(True)

pattern_avg = pattern_sum/ len(peaks)

pattern_supressed_sig = sig




# # try to suppress patterns
# fig, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 6))

# line1, = ax1.plot(t,sig, label = "og signal" )
# ax1.set_ylabel("Amplitude")
# ax1.set_xlabel("Time [s]")
# # ax1.set_title(f"Peak aligment correlation, peak @ {peaks_inSeconds[peak_of_interest_idx]} s")
# ax1.set_title(f"Pattern suppresion")
# ax1.grid(True)
# ax1.legend()

# for idx,ignoring in enumerate(peaks):
#     peak_of_interest_idx = idx
#     pattern_supressed_sig[peaks[peak_of_interest_idx]:peaks[peak_of_interest_idx]+win_size] -= pattern_avg

#     sample_time_range = np.arange(peaks[peak_of_interest_idx],peaks[peak_of_interest_idx]+win_size) /fs # range(peaks[0],peaks[0]+win_size) /fs
#     line2, = ax2.plot(sample_time_range,pattern_avg )


# ax2.set_ylabel("Amplitude")
# ax2.set_xlabel("Time [s]")
# ax2.grid(True)

# ax3.plot(t,pattern_supressed_sig,label = "pattern suppresed sig")
# ax3.set_ylabel("Amplitude")
# ax3.set_xlabel("Time [s]")
# ax3.grid(True)
# ax3.legend()



plt.show()