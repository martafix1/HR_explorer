import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class Track():
    staticID = 0
    def __init__(self, range_val, azi_val):
        self.ID = self.newID()
        
        # State estimation (Position and Velocity)
        self.r = range_val
        self.a = azi_val
        self.v_r = 0.0 # Range velocity
        self.v_a = 0.0 # Azimuth velocity
        
        # Track Management states
        self.miss_frames = 0
        self.alive_frames = 0
        self.state = "TENTATIVE" # TENTATIVE -> VALID -> DEAD
        
    def newID(self) -> int:
        Track.staticID += 1
        return Track.staticID - 1

    def predict(self):
        # Predict where the target should be in the next frame
        self.r_pred = self.r + self.v_r
        self.a_pred = self.a + self.v_a
        return np.array([self.r_pred, self.a_pred])

    def update(self, meas_r, meas_a, alpha=0.6, beta=0.3):
        # Alpha-Beta Filter Update
        # Alpha controls how much we trust the new measurement vs our prediction
        # Beta controls how fast we update our velocity
        
        # Residual (difference between measurement and prediction)
        res_r = meas_r - self.r_pred
        res_a = meas_a - self.a_pred
        
        # Update Position
        self.r = self.r_pred + alpha * res_r
        self.a = self.a_pred + alpha * res_a
        
        # Update Velocity
        self.v_r = self.v_r + beta * res_r
        self.v_a = self.v_a + beta * res_a
        
        self.miss_frames = 0
        self.alive_frames += 1
        if self.alive_frames > 3:
            self.state = "VALID"

def cluster_cfar_targets(targets, amplitudes):
    # DUMMY FUNCTION: Replace with actual clustering (e.g., DBSCAN)
    # Ideally, group adjacent points and return amplitude-weighted centroids.
    # For now, we'll just return the raw targets to keep the code running.
    return targets 



import numpy as np
import sys, os

import scipy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process
import processing.cfar as cfar

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


params = {  "DoA_azi_N_elements":8,
            "DoA_ele_N_elements":1,
            "DoA_azi_range_degs":90, "DoA_ele_range_degs":30,
            "Channel_processing":"DoA_customFFT",
            # "Channel_processing":"None",
            "Doppler_processing":"FFT",

            "range_index2dist"  : 0.046,
            "frame_index2time"  : 5e-2,
            "doppler_index2freq"  : 1/(5.76* 1e-3), # Active chirping time 5.76 ms
            "doppler_index2vel"  : 0.157, # Velocity resolution, m/s
            "radarRotation_deg" : 0
                   }




filePath = "data/unR_meas_noHR_32_rdr227_humancenter_06-03-2026_14-30-14.npz" 
# filePath = "data/unR_meas_noHR_33_rdr227_humancenterlowsitting_06-03-2026_14-33-03.npz"

filePath = "../VScodeSlozka/ros2-devcontainer-example-ws/DATA_UNrosed/unR_meas_21_michalauto_radarvodorovne_stani_24-04-2026_14-32-50.npz"


loadedData = loadNPZ(filePath)

frames = loadedData["frames"]

params = HR_process.defaultSliders(frames,params)



params["i_Frames_begin"] = 0
params["i_Frames_end"] = 300
params["i_Range_begin"] = 12
params["i_Range_end"] = 32

t0 = params["i_Frames_begin"] * params["frame_index2time"]
t1 = params["i_Frames_end"] * params["frame_index2time"]
t = np.linspace(t0,t1, params["i_Frames_end"] -params["i_Frames_begin"] )

# doppler_bin = 0
# ele_bin = 0
# azi_bin = 3
# range_bin = 25-  params["i_Range_begin"]

penteract, DoA_dict = HR_process.process_A(frames,params)



tracks = []
MAX_MISS_FRAMES = 5
GATE_DISTANCE = 3.0 # Maximum allowed distance to associate a track to a plot

history_cfar = []
history_tracks = []

# Main Loop
for frame in penteract[:,:,:,0,:]:
    # ... your CFAR logic ...
    frame_dopplerSum = np.abs(np.sum(frame,axis=0))  
    treshold = cfar.cell_max_CFAR_2D(data=frame_dopplerSum, guard_0=1, guard_1=0, train_0=2, train_1=0)
    
    targets_raMap = np.ones_like(treshold)
    targets_raMap[treshold>frame_dopplerSum] = 0
    raw_targets = np.argwhere(targets_raMap > 0)
    
    # 1. Plot Extraction (Clustering)
    amplitudes = frame_dopplerSum[raw_targets[:, 0], raw_targets[:, 1]]
    plots = cluster_cfar_targets(raw_targets, amplitudes)

    if not tracks:
        # Init Phase: create tentative tracks for all plots
        for p in plots:
            tracks.append(Track(p[0], p[1]))
        continue

    # 2. Predict step
    predicted_positions = np.array([t.predict() for t in tracks])

    # 3. Data Association (Hungarian Algorithm)
    if len(plots) > 0 and len(tracks) > 0:
        # Calculate distance matrix between all predictions and all plots
        cost_matrix = cdist(predicted_positions, plots)
        
        # Hungarian algorithm finds the optimal 1-to-1 assignment
        track_indices, plot_indices = linear_sum_assignment(cost_matrix)
        
        assigned_tracks = set()
        assigned_plots = set()
        
        # 4. Update Step
        for t_idx, p_idx in zip(track_indices, plot_indices):
            # Gating: Reject assignments that are too far away
            if cost_matrix[t_idx, p_idx] < GATE_DISTANCE:
                tracks[t_idx].update(plots[p_idx][0], plots[p_idx][1])
                assigned_tracks.add(t_idx)
                assigned_plots.add(p_idx)
    else:
        assigned_tracks = set()
        assigned_plots = set()

    # 5. Track Management (Misses and Deletions)
    for i, track in enumerate(tracks):
        if i not in assigned_tracks:
            track.miss_frames += 1
            
    # Remove dead tracks
    tracks = [t for t in tracks if t.miss_frames <= MAX_MISS_FRAMES]

    # 6. Track Initiation (New Tracks)
    for j, plot in enumerate(plots):
        if j not in assigned_plots:
            # Unassigned plot -> likely a new target
            tracks.append(Track(plot[0], plot[1]))

    # (Optional) Filter out tentative tracks when reporting to the UI/Next stage
    valid_tracks_to_report = [t for t in tracks if t.state == "VALID"]

    # 1. Log raw CFAR targets for background context
    if raw_targets.size > 0:
        history_cfar.append(raw_targets.copy())
    else:
        history_cfar.append(np.empty((0, 2)))

    # 2. Log current valid tracks
    current_frame_tracks = []
    for t in tracks:
        # Only log tracks that are somewhat established
        if t.state == "VALID" or t.alive_frames > 1:
            current_frame_tracks.append({
                'id': t.ID,
                'r': t.r,
                'a': t.a,
                'state': t.state
            })
    history_tracks.append(current_frame_tracks)



def visualize_tracker(history_cfar, history_tracks, tail_length=5):
    """
    history_cfar: List of numpy arrays [N, 2] containing (range, azi) for each frame.
    history_tracks: List of lists containing dictionaries of track states for each frame.
    """
    num_frames = len(history_cfar)
    if num_frames == 0:
        print("No data to visualize!")
        return

    # Setup the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2) # Make room for the slider
    
    # Set axis limits based on your data (Adjust these if your range bins go higher)
    max_range = max([np.max(c[:,0]) for c in history_cfar if c.size > 0] + [50]) 
    max_azi = 8 # As you mentioned, 8 azimuth bins
    
    ax.set_xlim(-1, max_azi)
    ax.set_ylim(-1, max_range + 5)
    ax.set_xlabel('Azimuth Bin')
    ax.set_ylabel('Range Bin')
    ax.set_title('Radar Tracker Playback')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Initialize plot objects (empty at first)
    cfar_scatter = ax.scatter([], [], c='lightgray', s=30, label='CFAR Detections', zorder=1)
    track_scatter = ax.scatter([], [], c='red', s=80, edgecolors='black', label='Active Tracks', zorder=3)
    
    # Lists to keep track of dynamic text and line objects so we can clear them
    texts = []
    tail_lines = []
    
    ax.legend(loc='upper right')

    # Slider Setup
    axframe = plt.axes([0.15, 0.05, 0.7, 0.03])
    sframe = Slider(axframe, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    def update(val):
        frame_idx = int(sframe.val)
        
        # 1. Update CFAR background
        cfar_data = history_cfar[frame_idx]
        if cfar_data.size > 0:
            # Note: matplotlib scatter expects (x, y) which is (azi, range)
            cfar_scatter.set_offsets(cfar_data[:, [1, 0]]) 
        else:
            cfar_scatter.set_offsets(np.empty((0, 2)))

        # 2. Update Tracks
        track_data = history_tracks[frame_idx]
        if len(track_data) > 0:
            track_coords = np.array([[t['a'], t['r']] for t in track_data])
            track_scatter.set_offsets(track_coords)
        else:
            track_scatter.set_offsets(np.empty((0, 2)))

        # 3. Clear old text IDs and Tails
        for txt in texts:
            txt.remove()
        texts.clear()
        
        for line in tail_lines:
            line.remove()
        tail_lines.clear()

        # 4. Draw new IDs and Tails
        for t in track_data:
            # Draw ID
            txt = ax.text(t['a'] + 0.2, t['r'] + 0.5, f"ID:{t['id']}", 
                          fontsize=9, color='darkred', zorder=4)
            texts.append(txt)
            
            # Draw Tail (look back 'tail_length' frames)
            tail_x, tail_y = [], []
            for back_idx in range(max(0, frame_idx - tail_length), frame_idx + 1):
                # Search for this track ID in past frames
                past_tracks = history_tracks[back_idx]
                for pt in past_tracks:
                    if pt['id'] == t['id']:
                        tail_x.append(pt['a'])
                        tail_y.append(pt['r'])
                        break
            
            if len(tail_x) > 1:
                line, = ax.plot(tail_x, tail_y, c='red', alpha=0.5, linewidth=2, zorder=2)
                tail_lines.append(line)

        fig.canvas.draw_idle()

    # Register the update function
    sframe.on_changed(update)
    
    # Initialize the first frame
    update(0)
    plt.show()

# --- Call this after your main loop ---
visualize_tracker(history_cfar, history_tracks, tail_length=5)