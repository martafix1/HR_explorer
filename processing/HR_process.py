import sys
import time
import processing.DoA as DoA
import numpy as np
import tqdm

sliders_allowedVals = { "i_Frames_begin":"!_+0INT","i_Frames_end":"!_+0INT",
                        "i_Range_begin":"!_+0INT","i_Range_end":"!_+0INT",
                        "i_Doppler_begin":"!_+0INT","i_Doppler_end":"!_+0INT",
                        "DoA_azi_N_elements":"!_+0INT",
                        "DoA_ele_N_elements":"!_+0INT",
                        "DoA_azi_range_degs":"!_+0FLOAT", "DoA_ele_range_degs":"!_+0FLOAT",
                        "Channel_processing":("DoA_customFFT","None"),
                        "Doppler_processing":("FFT","None")
                        }

sliders = { "i_Frames_begin":100,"i_Frames_end":600,
            "i_Range_begin":0,"i_Range_end":30,
            "i_Doppler_begin":0,"i_Doppler_end":32,
            "DoA_azi_N_elements":24,
            "DoA_ele_N_elements":12,
            "DoA_azi_range_degs":90, "DoA_ele_range_degs":30,
            
            "Channel_processing":"DoA_customFFT",
            "Doppler_processing":"None",
                   }

def defaultSliders(frames,sliders):
    n_og_frames,n_og_doppler,n_og_channel,n_og_range = frames.shape
    sliders["i_Frames_begin"] = 0
    sliders["i_Frames_end"] = n_og_frames

    sliders["i_Range_begin"] = 0
    sliders["i_Range_end"] = n_og_range

    sliders["i_Doppler_begin"] = 0
    sliders["i_Doppler_end"] = n_og_doppler

    return sliders



def process_A(frames,sliders):
    
    n_og_frames,n_og_doppler,n_og_channel,n_og_range = frames.shape
    print(f"Base shape: {(n_og_frames,n_og_doppler,n_og_channel,n_og_range)}  <-(N_Frames,N_Doppler,N_Channel,N_Range)")

    i_Frames_begin = sliders["i_Frames_begin"]
    i_Frames_end = sliders["i_Frames_end"]
    N_Frames = i_Frames_end - i_Frames_begin

    i_Range_begin = sliders["i_Range_begin"]
    i_Range_end = sliders["i_Range_end"]
    N_Range = i_Range_end - i_Range_begin

    i_Doppler_begin = sliders["i_Doppler_begin"]
    i_Doppler_end = sliders["i_Doppler_end"]
    N_Doppler = i_Doppler_end - i_Doppler_begin

    DoA_azi_N_elements = sliders["DoA_azi_N_elements"]
    DoA_ele_N_elements = sliders["DoA_ele_N_elements"]
    DoA_azi_range_degs = sliders["DoA_azi_range_degs"]
    DoA_ele_range_degs = sliders["DoA_ele_range_degs"]

    N_Azi = DoA_azi_N_elements
    N_Ele = DoA_ele_N_elements

    endFramesShape = (N_Frames,N_Doppler,N_Range,N_Ele,N_Azi)
    print(f"Resultant frame shape {endFramesShape} <-(N_Frames,N_Doppler,N_Range,N_Ele,N_Azi)")
    print(f"Coresponds to ranges {i_Frames_begin}:{i_Frames_end}, {i_Doppler_begin}:{i_Doppler_end}, {n_og_channel}, {i_Range_begin}:{i_Range_end} <-(N_Frames,N_Doppler,N_channel,N_Range)")
    
    
    doppler_processing = sliders["Doppler_processing"] # "FFT" #"raw","FFT"
    channel_processing = sliders["Channel_processing"] 

    #channel_processing = "DoA_customFFT" #"raw","DoA_customFFT" 

    # reduce frames
    frames_lockedFrames = frames[i_Frames_begin:i_Frames_end,:,:,:]
    
    # process range
    rangeData_lockedFrames = np.fft.fft(frames_lockedFrames,axis=3) # range fft
    # reduce ranges 
    rangeData_lockedFramesRanges = rangeData_lockedFrames[:,:,:,i_Range_begin:i_Range_end]

    # doppler processing
    if(doppler_processing == "FFT"):
        rangeDopplerData_lockedFramesRanges = np.fft.fft(rangeData_lockedFramesRanges,axis=1) # doppler fft
    else:
        rangeDopplerData_lockedFramesRanges = rangeData_lockedFramesRanges
    # reduce doppler
    rangeDopplerData_lockedFramesRangesDoppler = rangeDopplerData_lockedFramesRanges[:,i_Doppler_begin:i_Doppler_end,:,:]


    # channel processing (DoA)
    dataIN = rangeDopplerData_lockedFramesRangesDoppler.astype(np.complex64)
    doa_params = DoA.params
    doa_params["azi_resolution_N"] = DoA_azi_N_elements
    doa_params["ele_resolution_N"] = DoA_ele_N_elements
    doa_params["azimuth_range"] = DoA_azi_range_degs
    doa_params["elevation_range"] = DoA_ele_range_degs
    doa_dict = DoA.create_DFT_matrix(doa_params)
    

    if(channel_processing == "DoA_customFFT"):
        dataIN = rangeDopplerData_lockedFramesRangesDoppler
        pStart = time.perf_counter()

        sizeAprox = (N_Frames*N_Doppler*N_Range)*(N_Azi*N_Ele)*8 #size of A * size of B* datatype size, result in aprox Bytes
        sizeLimit = int(0.8e9) # approx Bytes size
        batch_num = (sizeAprox/sizeLimit)    
        batch_size = int(np.ceil(N_Frames/batch_num))  

        dataOUT = np.empty((N_Frames,N_Doppler,N_Range,int(N_Azi*N_Ele)),dtype=np.complex64)
        for i in tqdm.tqdm(range(0,N_Frames,batch_size)):
            i_end = min(i + batch_size, N_Frames)

            dataOUT[i:i_end,:,:,:] = np.tensordot(
            (dataIN[i:i_end,:,:,:] / (doa_dict["calib"][None, None, :, None])).astype(np.complex64),      # shape: (d0, d1, n_channels, d3)
            doa_dict["beam_vector_flat"].astype(np.complex64),    # shape: (n_channels, n_beams)
            axes=([2], [0])                  # contract over channel axis
            )


            pass
        
        dataOUT = dataOUT.reshape(
            N_Frames, N_Doppler, N_Range,
            len(doa_dict["ele_mesh_range"]),
            len(doa_dict["azi_mesh_range"])
        )
        pEnd = time.perf_counter()
        print(f"tensordot time: {pEnd-pStart}")
        
        # #print optimiziation - it was slower
        # path_info = np.einsum_path('abcd,cf->abdf', 
        #                     (dataIN / (doa_dict["calib"][None, None, :, None])).astype(np.complex64), 
        #                     doa_dict["beam_vector_flat"].astype(np.complex64),
        #                     optimize=True
        #                     )
        # print(path_info[1])

        # pStart = time.perf_counter()
        # dataOUT = np.einsum('abcd,cf->abdf', 
        #                     (dataIN / (doa_dict["calib"][None, None, :, None])).astype(np.complex64), 
        #                     doa_dict["beam_vector_flat"].astype(np.complex64),
        #                     optimize=True
        #                     )
        # dataOUT = dataOUT.reshape(
        #     N_Frames, N_Doppler, N_Range,
        #     len(doa_dict["ele_mesh_range"]),
        #     len(doa_dict["azi_mesh_range"])
        # )
        # dataOUT = np.abs(dataOUT)
        # pEnd = time.perf_counter()
        # print(f"Einsum time: {pEnd-pStart}")

    else:
        # move axis, put it all into azi vector
        # TODO tryout
        print("No DoA is likely wrong, work on it")
        dataOUT = np.moveaxis(dataIN,[2,3],[3,2])
        flat = np.resize(dataOUT[:,:,:,0], (N_Frames, N_Doppler, N_Range, doa_dict["ele_mesh_range"] * doa_dict["azi_mesh_range"]))
        dataOUT = flat.reshape(28, 15, 32,doa_dict["ele_mesh_range"] , doa_dict["azi_mesh_range"])
        
        
        
    
    penteract = dataOUT

    azi_mesh_points = doa_dict["azi_mesh_range"]
    ele_mesh_points = doa_dict["ele_mesh_range"]

    # penteract = Frames,Doppler,Range,Ele,Azi
    return penteract, doa_dict





if __name__ == "__main__":
    from FileIO.loadNPZ import loadNPZ
    filePath = "Projects/TestProject/analysis_modules/plot_radar_ampl/data/unR_meas_noHR_32_rdr227_humancenter_06-03-2026_14-30-14.npz"
    data = loadNPZ(filePath)
    frames : np.ndarray = data["frames"] 
    sliders = defaultSliders(frames,sliders)
    process_A(frames,sliders)