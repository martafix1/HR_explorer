
import numpy as np
import scipy
import matplotlib.pyplot as plt

def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray))

# Example pattern functions (you can replace these!)
def ekg_like(x):
    """Simple EKG-like spike"""
    return np.exp(-50 * (x - 0.2)**2) - 0.5*np.exp(-80 * (x - 0.25)**2)

def trapezoid_smooth(phase,flatDuty = 0.5, initPhase=0.0, lowPass_freq = 0.1,fs = 20):
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
    
   
    instantaneous_f_br = 0.3 + (trapezoid_smooth(t/20,initPhase=0.75,lowPass_freq=0.2,fs=fs) * 0.15)-0.05
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

    instantaneous_f_hr = 1 + (trapezoid_smooth(t/40,initPhase=0.75,fs=fs) * 0.2)-0.05

    pulse_duration = 0.4
    viablePulseDuration = 0.5 # it makes a pulse and then it goes to zero forever after roughly 0.5-0.6s 

    instantaneous_phase_cycles =  np.cumsum(instantaneous_f_hr) / fs 

    instantaneous_phase_cycles = instantaneous_phase_cycles % 1

    hr_sig = ekg_like(instantaneous_phase_cycles)/0.65 # to normilize it to 1

    return hr_sig, instantaneous_f_hr



def generate_composite_signal(fs=20,T=60):
# Time axis
    fs = 20  # sampling frequency
    T = 60      # total duration (seconds)
    t = np.linspace(0, T, int(fs*T), endpoint=False)


    BreathSig,BR_instantf = breath_sig_trapezoidFreqVary(t,fs)
    HeartSig, HR_instantf = heart_sig_trapezoidFreqVary(t,fs)

    sig = (HeartSig*0.1) + BreathSig

    return t, sig, BR_instantf, HR_instantf