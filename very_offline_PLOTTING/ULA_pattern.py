import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_array_factor(N=8, d_lambda=0.5, beta_deg=0):
    # 1. Setup variables
    beta_rad = np.radians(beta_deg)
    theta = np.linspace(0, 2 * np.pi, 1000) # Iterated variable theta
    
    # 2. Calculate total phase psi
    psi = 2 * np.pi * d_lambda * np.cos(theta) + beta_rad
    
    # 3. Calculate Normalized Array Factor
    AF = np.zeros_like(psi)
    
    # Create a mask to prevent division by zero when psi is a multiple of 2*pi
    zero_mask = np.isclose(np.mod(psi, 2 * np.pi), 0)
    
    # Set maximum value at the main lobes
    AF[zero_mask] = 1.0 
    
    # Calculate the pattern for all other angles
    AF[~zero_mask] = np.abs(np.sin(N * psi[~zero_mask] / 2) / (N * np.sin(psi[~zero_mask] / 2)))

    # 4. Create the Matplotlib Figures
    fig = plt.figure(figsize=(12, 5))
    fig.canvas.manager.set_window_title('Uniform Linear Array Plotter')
    
    # --- Rectangular Plot ---
    ax1 = fig.add_subplot(121)
    ax1.plot(np.degrees(theta), AF, linewidth=1.5)
    ax1.set_title('Rectangular Plot (Linear Scale)')
    ax1.set_xlabel(r'Observation Angle $\theta$ (degrees)')
    ax1.set_ylabel('Normalized Array Factor |AF|')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # --- Polar Plot ---
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.plot(theta, AF, linewidth=1.5)
    ax2.set_title('Polar Plot')
    ax2.set_theta_zero_location("N") # Align 0 degrees with the vertical (Z-axis)
    ax2.set_theta_direction(-1)      # Set clockwise direction
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Main Title
    plt.suptitle(f'Uniform Linear Array\nN={N}, $d/\lambda$={d_lambda}, $\\beta$={beta_deg}$^\circ$', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_array_factor_listAndLog_capable(N=8, d_lambda=0.5, beta_deg=0, log_axis=False, limitTo180Deg = False):
    # 1. Standardize inputs: convert scalars to single-element lists
    N_list = [N] if not isinstance(N, (list, np.ndarray, tuple)) else list(N)
    d_list = [d_lambda] if not isinstance(d_lambda, (list, np.ndarray, tuple)) else list(d_lambda)
    beta_list = [beta_deg] if not isinstance(beta_deg, (list, np.ndarray, tuple)) else list(beta_deg)
    
    theta = np.linspace(0, 2 * np.pi, 1000) # Iterated variable theta
    
    # 2. Setup the Matplotlib Figures
    fig = plt.figure(figsize=(13, 6))
    fig.canvas.manager.set_window_title('Uniform Linear Array Multi-Plotter')
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='polar')
    
    # 3. Iterate over all combinations of parameters
    for n_val, d_val, b_val in itertools.product(N_list, d_list, beta_list):
        beta_rad = np.radians(b_val)
        
        # Calculate total phase psi
        psi = 2 * np.pi * d_val * np.cos(theta) + beta_rad
        
        # Calculate Normalized Array Factor
        AF = np.zeros_like(psi)
        zero_mask = np.isclose(np.mod(psi, 2 * np.pi), 0)
        AF[zero_mask] = 1.0 
        AF[~zero_mask] = np.abs(np.sin(n_val * psi[~zero_mask] / 2) / (n_val * np.sin(psi[~zero_mask] / 2)))
        
        # Handle scaling options
        if log_axis:
            # Floor at -40 dB to prevent log10(0) from breaking the plot boundaries
            AF_display = 20 * np.log10(np.clip(AF, 1e-2, 1.0))
        else:
            AF_display = AF
        
        if limitTo180Deg:
            mask = theta <= np.pi
            theta_display = theta[mask]
            AF_display_deglim = AF_display[mask]
        else:
            theta_display = theta
            AF_display_deglim = AF_display

        # Dynamically generate label strings
        label_str = f"N={n_val}, d={d_val}, $\\beta$={b_val :.2f}°"
        
        # Plot this iteration's case on both axes
        ax1.plot(np.degrees(theta_display), AF_display_deglim, linewidth=1.5, label=label_str)
        ax2.plot(theta, AF_display, linewidth=1.5, label=label_str)
        
    # 4. Format Layout and Limits
    if log_axis:
        y_min, y_max = -40, 0
        y_label = 'Normalized Array Factor (dB)'
        title_suffix = '(dB Scale)'
    else:
        y_min, y_max = 0, 1.05
        y_label = 'Normalized Array Factor |AF|'
        title_suffix = '(Linear Scale)'
        
    # --- Rectangular Plot Setup ---
    ax1.set_title(f'Rectangular Plot {title_suffix}')
    ax1.set_xlabel(r'Observation Angle $\theta$ (degrees)')
    ax1.set_ylabel(y_label)
    if limitTo180Deg:
        ax1.set_xlim(0, 180)
    else:
        ax1.set_xlim(0, 360)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, linestyle='--', alpha=0.7)
    leg = ax1.legend(fontsize=11)
    leg.set_draggable(True)
    
    # --- Polar Plot Setup ---
    ax2.set_title(f'Polar Plot {title_suffix}')
    ax2.set_theta_zero_location("N") # Align 0 degrees up (Z-axis)
    ax2.set_theta_direction(-1)      # Clockwise
    ax2.set_rlim(y_min, y_max)
    
    if log_axis:
        ax2.set_rticks([-40, -30, -20, -10, 0])
    else:
        ax2.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        
    plt.suptitle('Uniform Linear Array', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_complex_array_factor(N=8, d_lambda=0.5, beta_deg=0):
    # 1. Setup variables
    beta_rad = np.radians(beta_deg)
    # Using 2500 points so the scatter plot merges into a smooth, continuous curve
    theta = np.linspace(0, 2 * np.pi, 2500) 
    
    # 2. Calculate total phase psi
    psi = 2 * np.pi * d_lambda * np.cos(theta) + beta_rad
    
    # 3. Calculate Complex Array Factor (Sum of Phasors)
    # n is an array column from 0 to N-1, allowing fast vectorized addition
    n = np.arange(N)[:, np.newaxis] 
    AF = np.sum(np.exp(1j * n * psi), axis=0)
    
    # 4. Extract Magnitude and Phase
    mag = np.abs(AF)
    # np.angle returns [-pi, pi], use mod to map cleanly to [0, 2*pi]
    phase = np.mod(np.angle(AF), 2 * np.pi) 
    
    # 5. Create the Matplotlib Figures
    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title('Complex Array Factor Plotter')
    
    # Define the rainbow colormap and normalize it from 0 to 2*pi
    cmap = 'hsv' 
    norm = plt.Normalize(vmin=0, vmax=2*np.pi)
    
    # --- Rectangular Plot ---
    ax1 = fig.add_subplot(121)
    ax1.scatter(np.degrees(theta), mag, c=phase, cmap=cmap, norm=norm, s=4, edgecolors='none')
    ax1.set_title('Rectangular Plot (Magnitude vs $\\theta$)')
    ax1.set_xlabel(r'Observation Angle $\theta$ (degrees)')
    ax1.set_ylabel('Unnormalized Array Factor |AF|')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, N * 1.05)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # --- Polar Plot ---
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.scatter(theta, mag, c=phase, cmap=cmap, norm=norm, s=4, edgecolors='none')
    ax2.set_title('Polar Plot')
    ax2.set_theta_zero_location("N") # 0 degrees points up
    ax2.set_theta_direction(-1)      # Clockwise
    ax2.set_ylim(0, N * 1.05)
    
    # --- Shared Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.05)
    cbar.set_label('Array Factor Phase (radians)', rotation=270, labelpad=20)
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    # Main Title
    plt.suptitle(f'Complex Uniform Linear Array\nN={N} elements, $d/\lambda$={d_lambda}, $\\beta$={beta_deg}$^\circ$', fontsize=14)
    plt.show()

def plot_complex_array_factor_logPossible(N=8, d_lambda=0.5, beta_deg=0, log_axis=True):
    # 1. Setup variables
    beta_rad = np.radians(beta_deg)
    theta = np.linspace(0, 2 * np.pi, 10000) 
    
    # 2. Calculate Complex Array Factor (Sum of Phasors)
    psi = 2 * np.pi * d_lambda * np.cos(theta) + beta_rad
    n = np.arange(N)[:, np.newaxis] 
    AF = np.sum(np.exp(1j * n * psi), axis=0)
    
    # 3. Extract Magnitude and Phase
    mag = np.abs(AF)
    phase = np.mod(np.angle(AF), 2 * np.pi) 
    # phase = np.angle(AF)
    
    # 4. Handle Axis Scaling
    if log_axis:
        # Normalize to max possible value (N) so the peak is 0 dB
        mag_norm = mag / N
        # Floor at -40 dB to avoid log10(0) crashing into negative infinity
        mag_display = 20 * np.log10(np.clip(mag_norm, 1e-2, 1.0))
        y_min, y_max = -40, 0
        y_label = 'Normalized Magnitude (dB)'
        title_suffix = '(dB Scale, Normalized to 0dB)'
    else:
        mag_display = mag
        y_min, y_max = 0, N * 1.05
        y_label = 'Unnormalized Magnitude |AF|'
        title_suffix = '(Linear Scale)'

    # 5. Create the Matplotlib Figures
    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title('Complex Array Factor Plotter')
    
    cmap = 'hsv' 
    norm = plt.Normalize(vmin=0, vmax=2*np.pi)
    # norm = plt.Normalize(vmin=np.min(phase), vmax=np.max(phase))
    
    # --- Rectangular Plot ---
    ax1 = fig.add_subplot(121)
    ax1.scatter(np.degrees(theta), mag_display, c=phase, cmap=cmap, norm=norm, s=4, edgecolors='none')
    ax1.set_title(f'Rectangular Plot {title_suffix}')
    ax1.set_xlabel(r'Observation Angle $\theta$ (degrees)')
    ax1.set_ylabel(y_label)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # --- Polar Plot ---
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.scatter(theta, mag_display, c=phase, cmap=cmap, norm=norm, s=4, edgecolors='none')
    ax2.set_title(f'Polar Plot {title_suffix}')
    ax2.set_theta_zero_location("N") 
    ax2.set_theta_direction(-1)      
    ax2.set_rlim(y_min, y_max) # Matplotlib beautifully places the floor (e.g. -40dB) at the center
    
    if log_axis:
        ax2.set_rticks([-40, -30, -20, -10, 0])
    
    # --- Shared Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.05)
    cbar.set_label('Array Factor Phase (radians)', rotation=270, labelpad=20)
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    # cbar.set_ticks([-np.pi,0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # cbar.set_ticklabels([r'$-\pi$','0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

    plt.suptitle(f'Uniform Linear Array\nN={N} elements, $d/\lambda$={d_lambda}, $\\beta$={beta_deg}$^\circ$', fontsize=14)
    plt.show()


def inverse_experiment_signalWithPhaseProgressionOverThetaRange(N=8, d_lambda=0.5, signalPhaseProgressionNOTBETA_deg = 90 ):
    
    theta = np.linspace(0,  np.pi, 180*2)
    calc_beta = -2*np.pi * d_lambda * np.cos(theta)

    channels = np.arange(N) #0..N-1
    channels = channels[:, None] # make it column, theta is automatically row

    channels_phase_recieved = channels * np.deg2rad(signalPhaseProgressionNOTBETA_deg)
    
    channels_phase_correction = channels * calc_beta

    recieved_phasors = np.exp(1j * channels_phase_recieved)

    correction_phase_shift_phasors = (np.exp(1j * channels_phase_correction))
    result = recieved_phasors * correction_phase_shift_phasors
    resultInTheta = np.abs(np.sum(result,axis=0)) 
    
    # array factor
    beta_rad = np.deg2rad(-signalPhaseProgressionNOTBETA_deg) 
    # theta = np.linspace(0, 2 * np.pi, 1000) # Iterated variable theta
    # 2. Calculate total phase psi
    psi = 2 * np.pi * d_lambda * np.cos(theta) + beta_rad
    # 3. Calculate Normalized Array Factor
    AF_phases = psi * channels
    AF_phasors = np.exp(1j* AF_phases )
    AF = np.abs(np.sum(AF_phasors,axis=0)) 
    # AF = np.zeros_like(psi)
    # Create a mask to prevent division by zero when psi is a multiple of 2*pi
    # zero_mask = np.isclose(np.mod(psi, 2 * np.pi), 0)
    # # Set maximum value at the main lobes
    # AF[zero_mask] = 1.0 
    # # Calculate the pattern for all other angles
    # AF[~zero_mask] = np.abs(np.sin(N * psi[~zero_mask] / 2) / (np.sin(psi[~zero_mask] / 2)))


    fig = plt.figure(figsize=(12, 5))
    fig.canvas.manager.set_window_title('Uniform Linear Array Plotter')
    ax1 = fig.add_subplot(111)
    ax1.plot(np.degrees(theta), resultInTheta, linewidth=1.5, label = "signal magnitude")
    ax1.plot(np.degrees(theta), AF, linewidth=1.5,linestyle = "--", label = "Array factor")
    ax1.set_title('Rectangular Plot (Linear Scale)')
    ax1.set_xlabel(r'Observation Angle $\theta$ (degrees)')
    # ax1.set_ylabel('Normalized Array Factor |AF|')
    # ax1.set_xlim(0, 360)
    # ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    plt.tight_layout()
    plt.show()



# Run with log_axis=True to see side-lobes clearly
# plot_complex_array_factor_logPossible(N=8, d_lambda=0.5, beta_deg=0, log_axis=True)

# Execute the plotter with default values (Broadside array)
# plot_array_factor(N=8, d_lambda=0.5, beta_deg=0)

# plot_complex_array_factor(N=8, d_lambda=0.5, beta_deg=0)

# plot_complex_array_factor_logPossible(N=8, d_lambda=0.5, beta_deg=0, log_axis=True)

# DirectionsOfArrival = np.deg2rad(np.array([30,90,120])) 
# betas_deg = np.rad2deg(np.pi * -np.cos(DirectionsOfArrival))
# # betas_deg =  np.array([0,45,90,135])
# print(f"Betas: {betas_deg}")

# plot_array_factor_listAndLog_capable(N = 8, d_lambda=[0.5,1], beta_deg=0) # grating lobes

# plot_array_factor_listAndLog_capable(N = [8,12], d_lambda=0.5, beta_deg=0) # N means better directivity


# plot_array_factor_listAndLog_capable(N = 8, d_lambda=0.5, beta_deg=betas_deg.tolist(),log_axis=False,limitTo180Deg=True)

#beamsteer by change of d
# plot_array_factor_listAndLog_capable(N = 4, d_lambda=0.5, beta_deg=90,log_axis=False,limitTo180Deg=True)


inverse_experiment_signalWithPhaseProgressionOverThetaRange()