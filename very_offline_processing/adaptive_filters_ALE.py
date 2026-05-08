import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def lms_ale(x, delay, mu, order):
    """
    Adaptive Line Enhancer using the LMS algorithm.
    
    Parameters:
    x     : Input signal (1D array)
    delay : The decorrelation delay (Delta)
    mu    : Learning rate (step size)
    order : Number of filter taps (length of the adaptive filter)
    """
    n_samples = len(x)
    # Weights initialized to zero
    w = np.zeros(order)
    
    # Pre-allocate outputs
    periodic_output = np.zeros(n_samples)
    residual_output = np.zeros(n_samples)
    
    # Buffer for delayed signal (x[n - delay])
    # We need to look back 'delay + order' samples
    for n in range(delay + order, n_samples):
        # 1. Get the delayed input vector (regressor)
        # We take a window of size 'order' starting from n - delay
        x_delayed_vector = x[n - delay : n - delay - order : -1]
        
        # 2. Predict current signal (y = w^T * x_delayed)
        y = np.dot(w, x_delayed_vector)
        
        # 3. Calculate error (the residual)
        # Error = Current Input - Predicted Periodic Part
        e = x[n] - y
        
        # 4. Update weights (LMS update rule)
        w = w + 2 * mu * e * x_delayed_vector
        
        # Store results
        periodic_output[n] = y
        residual_output[n] = e
        
    return periodic_output, residual_output

# --- 1. Generate Synthetic Signal ---
fs = 20  # Sampling frequency
t = np.arange(1000) / fs
f0 = 0.3    # Fundamental frequency

# Fundamental + 2nd harmonic + 3rd harmonic
# We add a slight magnitude variation (modulation) to test robustness
mag_mod = 1 + 0.5 * np.sin(2 * np.pi * 2 * t) 
fundamental = 1.0 * np.sin(2 * np.pi * f0 * t)
h2 = 0.05 * np.sin(2 * np.pi * 2 * f0 * t) * mag_mod
h3 = 0.1 * np.sin(2 * np.pi * 3 * f0 * t)

clean_periodic = fundamental + h2 + h3
noise = np.random.normal(0, 0.5, len(t))
input_signal = clean_periodic + noise

# --- 2. Run the ALE ---
# Parameters to tune:
# mu: If too large, it diverges. If too small, it tracks slowly.
# delay: Must be large enough to decorrelate noise but shorter than signal cycles.
# order: Higher order allows for more complex harmonic tracking.
periodic, residual = lms_ale(input_signal, delay=7, mu=0.0001, order=10)

# --- 3. Visualization ---
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original Input (Signal + Harmonics + Noise)")
plt.plot(t[:200], input_signal[:200], color='gray', alpha=0.5, label='Raw')
plt.plot(t[:200], clean_periodic[:200], 'r--', label='Ground Truth (Periodic)')
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Extracted Periodic Component (Fundamental + Harmonics)")
plt.plot(t[:200], periodic[:200], color='blue')

plt.subplot(3, 1, 3)
plt.title("Residual (Noise / Removed Part)")
plt.plot(t[:200], residual[:200], color='green')

plt.tight_layout()
plt.show()