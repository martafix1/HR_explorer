import numpy as np
import scipy
import matplotlib.pyplot as plt
# rng = np.random.default_rng()


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

# sliding window autocorrelation
window_begin = 20
window_end = 100
win_size = window_end-window_begin
print(f"win size: {win_size/fs} [s]")

initial_pattern = sig[window_begin:window_end]
initial_pattern_timeBase = np.arange(0,win_size) #/fs lets keep it in samples


average_pattern = np.zeros(win_size*2)
average_pattern_timeBase = np.arange(0,win_size*2)


result = scipy.signal.correlate(sig, initial_pattern , mode='valid')

peaks, properties = scipy.signal.find_peaks(result, height=2, distance=2)

peaks_inSeconds = (peaks + win_size/2)/fs
print(peaks_inSeconds)
print(properties)



curve = scipy.interpolate.interp1d(initial_pattern_timeBase, initial_pattern, kind='cubic',bounds_error = True, fill_value="NaN")


# plt.plot(initial_pattern_timeBase,initial_pattern,label = "init pattern")
# plt.plot(initial_pattern_timeBase,curve(initial_pattern_timeBase),label = "its fit")
# plt.legend()
# plt.grid()
# plt.title("Init pattern and its fit")
# plt.show()

# #  alfa is inverse stretch factor, tau is negative time offset (a = 0.5, tau = -20) ~ signal is 2x longer and begins 20 frames later
# def model_stretch(params,t_pattern, t_req, curve):
#     alpha, tau  = params
#    
#     pattern_tLims = [t_pattern[0],t_pattern[-1]]    # [0, N-1]
#
#     # newT = alpha * t_curve + tau
#     # model_Y = curve(newT)
#     # return model_Y

def model_stretch_TimeOfInterest(params,t_pattern):
    alpha, tau  = params
    
    pattern_tLims = [t_pattern[0],t_pattern[-1]]    # [0, N-1]

    ToI = pattern_tLims+tau # time of interest
    ToI[1] *= alpha
    return ToI


def residuals_stretch_dualInterpolation(params, y_pattern, t_pattern,ref_curve,t_ref):
    alpha, tau  = params

    ToI = model_stretch_TimeOfInterest(params,t_pattern)
    ref_tLims = [t_ref[0],t_ref[-1]]                        # [0-W, M-1-W ]

    print(f"  {t_ref[0] :.1f} -|- {ToI[0] :.2f}-{ToI[1]:.2f} -|- {t_ref[-1] :.1f} {ToI[1]-ToI[0] :.3f}  # Params| alpha: {alpha:.4f}, tau: {tau:.4f}")

    if (any(ToI < ref_tLims[0]) or any(ref_tLims[1] < ToI)): # make sure we stay within the support of reference, it should be big enough
        print(f"LSM needs bigger reference support!")
        return np.ones_like(t_pattern)*1e6

    support_pattern = t_pattern
    support_ref = np.linspace(ToI[0],ToI[1],len(support_pattern))
    

    y_ref = ref_curve(support_ref)
    y_pattern = y_pattern

    diff = y_ref - y_pattern
    return diff
    

def model_stretch__Pattern_over_Ref(params,t_ref,curve_pattern,t_pattern):

    alpha,tau = params
    ToI = model_stretch_TimeOfInterest(params,t_pattern)
    
    mask = (t_ref >= ToI[0]) & (t_ref <= ToI[1])

    pattern_support_inRefTimeDomain = t_ref[mask]
    pattern_support = (pattern_support_inRefTimeDomain - tau)/alpha

    y_pattern = curve_pattern(pattern_support)

    return pattern_support_inRefTimeDomain, y_pattern
    


# curve begins at t_ref = 0
def residuals_stretch(params, t_ref, y_ref,t_curve,curve):
    alpha, tau  = params
    t_curve_edges = np.array([t_curve[0],t_curve[-1]] ) # try offset of 1 so it does not land in the zero region  # for this to work alfa must be positive
    
    curveLen_og = len(t_curve) 

    t_ofInterest_common = (t_curve_edges - tau) / alpha
    
    t_min = t_ofInterest_common[0]
    t_max = t_ofInterest_common[1]

    # timing issues: ref and curve dont have same duration. We need to calculate the diff only for valid samples, the rest is curve zeropad
    # if we were to calculate only the valid samples, LSM falls appart as it cannot keep its jacobian valid
    # solution might be to allways return the biggest possible vector = y_ref support, but mask the invalid values to zeros. 
    # that might still fuck up the jacobian but hopefully less so. 


    mask = (t_ref >= t_min) & (t_ref <= t_max)
    mask_invalid = np.invert(mask)
    # y_selected = y_ref[mask]
    # t_selected = t_ref[mask] #this needs to be translated back into curve timebase
    # t_selected_curve = t_selected + t_ref[0] 
    # t_curve_refSupport = t_ref

    # calculate the model output for full ref support. t_ref has zero already at the correct place
    y_model = model_stretch(params, t_ref[mask] ,curve)
    diff = np.zeros_like(y_ref)
    diff[mask] = y_model - y_ref[mask] # vector caluclation on the full ref support
    # diff[mask_invalid] = 0
    
    # diff *=  np.count_nonzero(mask)/curveLen_og # linearly increase cost for each sample unused (reduce for each other sample)
    
    print(f"ToI| {t_min :.2f}-{t_max:.2f} ~ {t_max-t_min :.3f} # {np.count_nonzero(mask)}  # Params| alpha: {alpha:.4f}, tau: {tau:.4f}")
    # plt.plot(t_ref,y_ref,label = "reference")
    # plt.plot(t_ref[mask],y_model[mask],label = "model")
    # plt.plot(t_ref,diff,label = "diff")
    # plt.legend()
    # plt.show()

    return  diff


win_stretch = 30

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
ax1.plot(t*fs,sig, label = "ref signal")
ax1.set_ylabel("Amplitude")
# ax1.set_xlabel("Time [s]")
# ax1.set_title("CrossCorrelation")
ax1.grid(True)
ax1.legend()

for i, peak_oI in enumerate(peaks):
    
    t_pattern = initial_pattern_timeBase
    y_pattern = initial_pattern

    win_stretch_local = win_stretch
    if (peak_oI - win_stretch) < 0:
        win_stretch_local = win_stretch + (peak_oI - win_stretch)
        # make windows smaller for a signle turn 

    t_ref = np.arange(0-win_stretch_local,win_size+win_stretch_local)
    y_ref = sig[ (peak_oI - win_stretch_local): (peak_oI+win_size+win_stretch_local)] 
    curve_ref = scipy.interpolate.interp1d(t_ref, y_ref, kind='cubic',bounds_error = True, fill_value="NaN")
    #####      alpha, tau
    x0      = [ 1.0 , 0.0] 
    bounds  = ([0.3 ,-win_size ], #min
            [3   , win_size ]) #max
    # (params, y_pattern, t_pattern,ref_curve,t_ref)
    res = scipy.optimize.least_squares(residuals_stretch_dualInterpolation, x0,bounds=bounds, args=(y_pattern, t_pattern,curve_ref,t_ref))
    alpha,tau = res.x
    print(f" a: {alpha}, tau {tau/fs} [s]")


    plt.figure(2)
    plt.subplot(2,1,1)
    ToI = model_stretch_TimeOfInterest((alpha,tau),t_pattern)
    
    support_ref = np.linspace(ToI[0],ToI[1],len(t_pattern))
    y_ref = curve_ref(support_ref)
    
    plt.plot(support_ref,y_ref,label = "ref")
    plt.plot(t_pattern,y_pattern,label = "pattern")
    plt.legend()

    plt.subplot(2,1,2)
    y_ref = sig[ (peak_oI - win_stretch_local): (peak_oI+win_size+win_stretch_local)] 
    plt.plot(t_ref,y_ref,label="ref")

    t_pattern_plot, y_pattern_plot = model_stretch__Pattern_over_Ref((alpha,tau),t_ref, curve ,t_pattern)
    plt.plot(t_pattern_plot,y_pattern_plot,label="model")

    plt.legend()
    plt.show()

    # t_curve_edges = np.array([initial_pattern_timeBase[0],initial_pattern_timeBase[-1]] ) 
    # t_ofInterest_common = (t_curve_edges - tau) / alpha

    # t_min = t_ofInterest_common[0]
    # t_max = t_ofInterest_common[1]
    # mask = (t_ref >= t_min) & (t_ref <= t_max)

    # pattern_sample = y_ref[mask] #take the actuall pattern sample.
    # pattern_sample_timebase = np.arange(0,len(pattern_sample))
    # curve2 = scipy.interpolate.interp1d(pattern_sample_timebase, pattern_sample, kind='cubic',bounds_error = True, fill_value="NaN")

    # N = len(pattern_sample)
    # M = len(average_pattern)
    # k = (N-1)/(M-1)

    # average_pattern += curve2(average_pattern_timeBase*k)

    # ax2.plot(peak_oI+pattern_sample_timebase,pattern_sample)
    
ax2.grid(True)
plt.show()