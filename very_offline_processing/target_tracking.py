
import numpy as np
import sys, os

import scipy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FileIO.loadNPZ import loadNPZ
import processing.HR_process as HR_process
import processing.cfar as cfar

import matplotlib.pyplot as plt


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

# data_new = penteract[:,doppler_bin,range_bin,ele_bin,azi_bin]
# phase = np.angle(data_new)
# magnitude = np.abs(data_new)
# phase_unwrp = np.unwrap(phase)


# tracks 
class Track():
    staticID = 0
    def __init__(self,range,azi) -> None:
        self.range = range #None | int
        self.azi = azi #None | int
        self.ID = self.newID() # None | int
        self.miss_frames = 0
        self.grace_frames = 0
        self.alive_frames = 0
    
    def newID(self) -> int:
        Track.staticID +=1
        return Track.staticID-1
        




validTracks =  []
tennativeTracks =  []
updated_tracks_idxs = []

initPhase = True

for frame in penteract[:,:,:,0,:]:
    # frame = [dopp,range,azi]
    frame_dopplerSum = np.abs(np.sum(frame,axis=0))  # [range,azi]
    treshold = cfar.cell_max_CFAR_2D(data=frame_dopplerSum, guard_0=1, guard_1=0, train_0=2, train_1=0)
    targets_raMap = np.ones_like(treshold)
    targets_raMap[treshold>frame_dopplerSum] = 0

    targets = np.argwhere(targets_raMap > 0) # ordered from range 0 to end as major, then azi 0 to end as minor

    if initPhase:

        for target in targets:
            range_, azi = target
            validTracks.append(Track(range_,azi) )

        initPhase = False

    else:
        
        # asociate tracks to cfar targets
        for vTrack in validTracks:

            range_ = vTrack.range
            azi = vTrack.azi
            
            # possible targets - NN kinda
            azi_expand = 1
            r_expand = 1
            possibleTargets =   targets[(targets[:,0] >= range_-r_expand) & (targets[:,0] <= range_+r_expand) &
                                        (targets[:,1] >= azi-azi_expand) & (targets[:,1] <= azi+azi_expand)]
            if possibleTargets.size > 0:
                pTarget_amplitudes = frame_dopplerSum[possibleTargets[:, 0], possibleTargets[:, 1]]

                selected_target = np.argmax(pTarget_amplitudes)

                new_range = possibleTargets[selected_target,0]
                new_azi = possibleTargets[selected_target,1]
                vTrack.range = new_range
                vTrack.azi = new_azi
            else:
                vTrack.miss_frames +=1
            # possibleTargets = targets[]

            if vTrack.miss_frames > 0:
                vTrack.grace_frames +=1

            # remove in another loop.
            if vTrack.grace_frames > 5:
                pass
            pass

            



        pass


