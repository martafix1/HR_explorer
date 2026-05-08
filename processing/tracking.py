import numpy as np
import sys, os, time
import copy
import scipy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# from FileIO.loadNPZ import loadNPZ
# import processing.HR_process as HR_process
import processing.cfar as cfar




def clustering_touchingNeighbours(raw_targets, amplitudes):

    targets = {tuple(pt) for pt in raw_targets}  # set of (r, a)
    amp_map = {tuple(pt): amp for pt, amp in zip(raw_targets, amplitudes)}

    pattern = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

    plots = []

    while targets:
        # 1. Find strongest target
        TopTarget = max(targets, key=lambda pt: amp_map[pt])

        cluster_points = []
        cluster_amps = []

        # 2. Check neighbors based on pattern
        for dr, da in pattern:
            neighbor = (TopTarget[0] + dr, TopTarget[1] + da)

            if neighbor in targets:
                cluster_points.append(neighbor)
                cluster_amps.append(amp_map[neighbor])

        # Convert to numpy
        cluster_points = np.array(cluster_points)
        cluster_amps = np.array(cluster_amps)

        # Sort by amplitude first
        sort_idx = np.argsort(cluster_amps)[::-1] # Descending
        cluster_points = cluster_points[sort_idx]
        cluster_amps = cluster_amps[sort_idx]

        # 3. Compute weighted centroid
        centroid = np.average(cluster_points, axis=0, weights=cluster_amps)

        # 4. Remove used targets
        for pt in cluster_points:
            targets.remove(tuple(pt))

        # 5. Store result
        plots.append({
            "centroid": centroid,
            "points": cluster_points,
            "amplitude_sum": cluster_amps.sum()
        })

    return plots



def _compute_weighted_cost(tracks, detections, weights=(1.0, 1.0)):
    tracks = np.asarray(tracks)        # (N, 2)
    detections = np.asarray(detections)  # (M, 2)
    w = np.asarray(weights)            # (2,)

    # diff [i,j,:] = tracks[i] - detections[j]   # if the detections & tracks were 1D it would be 2D, this is to acomodate the higher dim
    diff = tracks[:, None, :] - detections[None, :, :]
    weighted = diff * w  # elementwise scaling per dimension
    cost = np.linalg.norm(weighted, axis=2)

    return cost

def _apply_gating(cost_matrix, max_distance):
    gated_cost = cost_matrix.copy()
    gated_cost[gated_cost > max_distance] = 1e9  # effectively "no match"
    return gated_cost


def globalNN_assignment(tracks,detections,tracking_params):
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
     
    w_range = tracking_params["match W_range"]
    w_azi = tracking_params["match W_azi"]
    gate_dist = tracking_params["gating dist"]
    cost_matrix = _compute_weighted_cost(tracks=tracks,detections=detections,weights=(w_range,w_azi)) # norm 2 = with squareroot! might be slower
    cost_matrix_gated = _apply_gating(cost_matrix,max_distance=gate_dist)

    track_idxs, detection_idxs = scipy.optimize.linear_sum_assignment(cost_matrix_gated)

    matches = [] #
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for t, d in zip(track_idxs, detection_idxs):
        if cost_matrix_gated[t, d] <= gate_dist:
            matches.append((t, d))
            unmatched_tracks.remove(t)
            unmatched_detections.remove(d)
    return matches, unmatched_tracks, unmatched_detections


def extract_track_signals(frame_data, detection, merge_bins=True):
    """
    frame_data: shape (Doppler, Range, Azi)
    detection: dict containing 'points' and 'amplitudes' sorted by strength
    """
    pts = detection["points"]
    # amps = detection["amplitudes"]

    # Combine two strongest bins or just the strongest one
    if merge_bins and len(pts) >= 2:
        pt1, pt2 = pts[0], pts[1]
        
        # w1, w2 = amps[0], amps[1]  # not needed coz we merge complex data - the phase will already be merged based on the amplitudes
        # w_sum = w1 + w2
        # w1, w2 = w1/w_sum, w2/w_sum
        
        # Linearly combine the complex signals
        sig1 = frame_data[:, pt1[0], pt1[1]]
        sig2 = frame_data[:, pt2[0], pt2[1]]
        sig_merged = (sig1) + (sig2)
    else:
        pt1 = pts[0]
        sig_merged = frame_data[:, pt1[0], pt1[1]]

    # Power across all doppler bands
    pow_all = np.sum(np.abs(sig_merged)**2)
    # Power for doppler[1:end] - only "fast" movements
    pow_high_dop = np.sum(np.abs(sig_merged[1:])**2)
    
    # Complex value of Doppler bin 0 for phase unwrapping
    complex_d0 = sig_merged[0]

    return pow_all, pow_high_dop, complex_d0


def finalize_signal_logs(track_logs, decay_alpha=0.2):
    """
    Converts lists to numpy arrays, applies phase unwrapping, 
    and calculates the slow decay on powers.
    """
    finalized_data = {}
    
    # Simple Exponential Moving Average (EMA) for the decay
    def apply_decay(arr, alpha):
        out = np.zeros_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out

    for tid, log in track_logs.items():
        frames = np.array(log["frames"])
        p_all = np.array(log["pow_all"])
        p_high = np.array(log["pow_high"])
        comp_d0 = np.array(log["comp_d0"])
        
        # TASK 2: Efficient Phase Unwrapping
        # Because we merged the complex baseband signals before taking the angle,
        # the phase will be mathematically stable. np.unwrap handles the 2*pi jumps.
        phases_raw = np.angle(comp_d0)
        phases_unwrapped = np.unwrap(phases_raw)
        
        # TASK 3: Post-process decay (makes tuning extremely fast later)
        p_all_decay = apply_decay(p_all, decay_alpha)
        p_high_decay = apply_decay(p_high, decay_alpha)
        
        finalized_data[tid] = {
            "frames": frames,
            "power_all": p_all,
            "power_all_decay": p_all_decay,
            "power_high_dop": p_high,
            "power_high_dop_decay": p_high_decay,
            "phase_raw": phases_raw,
            "phase_unwrapped": phases_unwrapped
        }
        
    return finalized_data


# # tracking interface
# tracking_history = [
#     frame_0_tracks,
#     frame_1_tracks,
#     frame_2_tracks,
#     ...
# ]
#  Where each is:
# frame_k_tracks = [
#     track_obj_1,
#     track_obj_2,
#     ...
# ]
# Where each is:
# track_obj = {
#     "id": int,        # unique track ID (persistent across frames)
#     "r_bin": int,     # range bin index
#     "azi_bin": int    # azimuth bin index
# }


def track_allData(penteract, cfar_params, tracking_params):
    ID = 0
    tracking_history = []
    track_signal_logs = {}

    frame_idx = 0
    def log_track_data(t_id, det_idx):
        pow_all, pow_high, comp_d0 = extract_track_signals(frame, detections[det_idx], merge_bins=tracking_params["Sig. extr: merge bins"])
        if t_id not in track_signal_logs:
            track_signal_logs[t_id] = {"frames": [], "pow_all": [], "pow_high": [], "comp_d0": []}
        
        track_signal_logs[t_id]["frames"].append(frame_idx)
        track_signal_logs[t_id]["pow_all"].append(pow_all)
        track_signal_logs[t_id]["pow_high"].append(pow_high)
        track_signal_logs[t_id]["comp_d0"].append(comp_d0)

    time_TrackingStart = time.perf_counter()
    time_cfar_sum = 0
    time_clustering_sum = 0
    time_assignment_sum = 0
    time_maintnance_sum = 0
    print(f"Tracking launched")
    dopp0_w, doppHigh_w = cfar_params["dopp0 weight"],cfar_params["dopp>0 weight"]
    for frame_idx, frame in enumerate(penteract[:,:,:,0,:]):
        time_cfar_start = time.perf_counter()
        frame[0,:,:] *= dopp0_w
        frame[1:,:,:] *=  doppHigh_w
        frame_dopplerSum = np.abs(np.sum(frame,axis=0))  

        if cfar_params["Method"] == "CA 2D CFAR":
            treshold = cfar.cell_average_CFAR_2D(frame_dopplerSum,cfar_params["guard range"],cfar_params["guard azi"],cfar_params["train range"],cfar_params["train azi"],dim0_i=0,dim1_i=1)
        elif cfar_params["Method"] == "cMax 2D CFAR":
            treshold = cfar.cell_max_CFAR_2D(frame_dopplerSum,cfar_params["guard range"],cfar_params["guard azi"],cfar_params["train range"],cfar_params["train azi"])
        else:
            treshold = frame_dopplerSum  
            print("Unknown CFAR method")
        
        treshold *= cfar_params["treshold scale"]
        
        targets_raMap = np.ones_like(treshold)
        targets_raMap[treshold>frame_dopplerSum] = 0
        raw_targets = np.argwhere(targets_raMap > 0)
        
        time_cfar_stop = time.perf_counter()
        time_cfar_sum += time_cfar_stop-time_cfar_start

        # 1. Plot Extraction (Clustering)
        amplitudes = frame_dopplerSum[raw_targets[:, 0], raw_targets[:, 1]]
        detections = clustering_touchingNeighbours(raw_targets, amplitudes)
        time_clustering_stop = time.perf_counter()
        time_clustering_sum += time_clustering_stop - time_cfar_stop
        
        # Prediction step - dynamics as in "it was there last time i saw it"
        if len(tracking_history) > 0:
            tracks_pred = [track["centroid"] for track in tracking_history[-1]] 
        else:
            tracks_pred = []
        
        detections_centroid = [detection["centroid"] for detection in detections]
        # Track assignment - find which tracks match and which do not 
        
        matches, unmatched_tracks, unmatched_detections = globalNN_assignment(tracks=tracks_pred,detections=detections_centroid, tracking_params= tracking_params)
        
        time_assignment_stop = time.perf_counter()
        time_assignment_sum += time_assignment_stop - time_clustering_stop
        # Track maintnaince 
        stability_miss = 5 # max misses in grace period
        stability_grace = 8 
        tentative_dur = 3
        
        
        next_tracks = []
        ## Already existing track association
        if len(matches)>0:
            previous_tracks = copy.deepcopy(tracking_history[-1])
            for match in matches:
                previous_tracks[match[0]]["r_bin"] = int(detections[match[1]]["centroid"][0])
                previous_tracks[match[0]]["azi_bin"] = int(detections[match[1]]["centroid"][1])
                previous_tracks[match[0]]["centroid"] = detections[match[1]]["centroid"]

                previous_tracks[match[0]]["alive"] +=1
                

                if(previous_tracks[match[0]]["status"] == "TENTATIVE"):
                    if previous_tracks[match[0]]["alive"] > tentative_dur:
                        previous_tracks[match[0]]["status"] = "VALID"
                elif(previous_tracks[match[0]]["status"] == "UNSTABLE"):
                    longerSurviveLonger =  1
                    if(previous_tracks[match[0]]["alive"] > 60):
                        longerSurviveLonger = 2
                    previous_tracks[match[0]]["stability m/t"] += np.array([0,1]) # increment just the frames
                    # if more than 3 misses in more than 5 frames
                    if previous_tracks[match[0]]["stability m/t"][1] > stability_grace * longerSurviveLonger and previous_tracks[match[0]]["stability m/t"][0] > stability_miss*longerSurviveLonger:
                        continue 
                    elif previous_tracks[match[0]]["stability m/t"][1] > stability_grace:
                        previous_tracks[match[0]]["status"] = "VALID" # back to valid

                # log existing track - coz it survived the selection process
                log_track_data(previous_tracks[match[0]]["id"], match[1]) 
                next_tracks.append(previous_tracks[match[0]])
                pass
        
        ## Check on lost tracks
        if len(unmatched_tracks) >0:
            previous_tracks = copy.deepcopy(tracking_history[-1])
            for t_idx in unmatched_tracks:
                if previous_tracks[t_idx]["status"] == "TENTATIVE": #if tentative let it die
                    continue
                elif previous_tracks[t_idx]["status"] == "UNSTABLE":
                    previous_tracks[t_idx]["stability m/t"] += 1 # increment both
                    longerSurviveLonger =  1
                    if(previous_tracks[t_idx]["alive"] > 60):
                        longerSurviveLonger = 2
                    # if more than 3 misses in more than 5 frames
                    if previous_tracks[t_idx]["stability m/t"][1] > stability_grace * longerSurviveLonger and previous_tracks[t_idx]["stability m/t"][0] > stability_miss*longerSurviveLonger:
                        continue
                else:
                    previous_tracks[t_idx]["status"] = "UNSTABLE"
                    previous_tracks[t_idx]["stability m/t"] = np.array([1,1])
                
                previous_tracks[t_idx]["alive"] +=1
                # cannot log radar values here coz there is no data
                next_tracks.append(previous_tracks[t_idx])


        ## Create new tracks
        if len(unmatched_detections) >0:
            for d_idx in unmatched_detections:
                track = {
                "id": ID,        
                "r_bin": int(detections[d_idx]["centroid"][0]),     
                "azi_bin": int(detections[d_idx]["centroid"][0]),
                "centroid": detections[d_idx]["centroid"],
                "status": "TENTATIVE",
                "alive" : 1
                } 
                log_track_data(ID, d_idx)
                ID +=1
                next_tracks.append(track)

        
        # Save outputs
        tracking_history.append(next_tracks)
        time_maintnance_stop = time.perf_counter()
        time_maintnance_sum += time_maintnance_stop - time_assignment_stop


        # test clustering
        # for plot in detections:
        #     track = {
        #         "id": ID,        
        #         "r_bin": int(plot["centroid"][0]),     
        #         "azi_bin": int(plot["centroid"][1]),
        #         "centroid": plot["centroid"]    
        #     } 
        #     ID +=1
        #     tracks.append(track)
        # Save outputs
        # tracking_history.append(track)
    
    time_postprocessing_start = time.perf_counter()
    final_signals = finalize_signal_logs(track_signal_logs, decay_alpha=0.15)
    time_TrackingEnd = time.perf_counter()
    print(f"Tracking - total time: {time_TrackingEnd- time_TrackingStart:.3f} s")
    print(f"CFAR: {time_cfar_sum:.3f}")
    print(f"Clustering: {time_clustering_sum:.3f}")
    print(f"Assignment: {time_assignment_sum:.3f}")
    print(f"Maintnaince: {time_maintnance_sum:.3f}")
    print(f"Finalizing: {time_TrackingEnd-time_postprocessing_start :.3f}")

    return tracking_history, final_signals
    pass

    

if __name__ == "__main__":
    tracks = [[2,0],
              [4,2],
              [2,1],
              ]
    detections = [[3,0],
                  [4,1]]
    
    max_distance = 3
    cost_matrix = _compute_weighted_cost(tracks=tracks,detections=detections,weights=(1,2))
    cost_matrix_gated = _apply_gating(cost_matrix,max_distance=3)

    track_idxs, detection_idxs = scipy.optimize.linear_sum_assignment(cost_matrix_gated)

    matches = [] #
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for t, d in zip(track_idxs, detection_idxs):
        if cost_matrix_gated[t, d] <= max_distance:
            matches.append((t, d))
            unmatched_tracks.remove(t)
            unmatched_detections.remove(d)


    print("kek")