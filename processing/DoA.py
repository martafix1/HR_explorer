

import numpy as np


params = {  'antenna_array_x'   : [  0.0, 0.5, 1.0, 1.5, 1.0, 1.5, 2.0, 2.5, 2.0, 2.5, 3.0, 3.5],
            'antenna_array_y'   : [  0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            'azimuth_range'     : 90,
            'elevation_range'   : 30, #15,
            # "azimuth_step"      : 22.5,
            # "elevation_step"    : 15,
            "azi_resolution_N"  : 64,
            "ele_resolution_N"  : 32,
            "tx_channels"       : [0,1,2],
            "rx_channels"       : [0,1,2,3],
            "n_rx"              : 4,
            "n_tx"              : 3,
            "n_channels"        : 12,
}


# create channel index array
channel_ind = [0]*len(params["tx_channels"])*len(params["rx_channels"])
i = 0
for tx in params["tx_channels"]:
    for rx in params["rx_channels"]:
        channel_ind[i] = int((tx*params["n_rx"])+rx)
        i+=1

params["channel_ind"] = channel_ind
print(channel_ind)




calib_wrongtype = [0.43033, -0.90267, 0.42070, -0.90720, 0.20025, -0.97974, 0.19226, -0.98134, 0.18629, -0.98249, 0.16594, -0.98614, -0.06994, -0.99755, -0.06332, -0.99799, 0.94481, -0.32763, 0.93498, -0.35469, 0.83448, -0.55105, 0.82928, -0.55883]
calib = np.array(calib_wrongtype,dtype=np.float32).view(np.complex64).copy()
# print(calib)

params["doa_offset_calibration_array"] = calib


def create_DFT_matrix(params: dict) : 
    antenna_position_x = np.array(params["antenna_array_x"])[params["channel_ind"]]
    antenna_position_y = np.array(params["antenna_array_y"])[params["channel_ind"]]
    # print(antenna_position_x)
    # print(antenna_position_y)

    # azi_mesh_range = np.arange(-params["azimuth_range"], params["azimuth_range"]+params["azimuth_step"], params["azimuth_step"])
    # ele_mesh_range  = np.arange(-params["elevation_range"], params["elevation_range"]+params["elevation_step"], params["elevation_step"])

    azi_mesh_range =np.linspace(start= -params["azimuth_range"], stop= params["azimuth_range"],num= params["azi_resolution_N"] )
    ele_mesh_range = np.linspace(start= -params["elevation_range"], stop= params["elevation_range"],num= params["ele_resolution_N"] )

    #print(azi_mesh_range[1:]-azi_mesh_range[:-1])


    # azi_mesh_range = np.arange(-params["azimuth_range"], params["azimuth_range"]+params["azimuth_step"], params["azimuth_step"])
    # ele_mesh_range  = np.arange(-params["elevation_range"], params["elevation_range"]+params["elevation_step"], params["elevation_step"])


    azi_mesh, ele_mesh = np.meshgrid(azi_mesh_range, ele_mesh_range, indexing='xy')

    # print(f"azi_mesh: {azi_mesh}" )
    # print(f"ele_mesh: {ele_mesh}" )

    azi_mesh_vector = np.tile(azi_mesh, [params["n_channels"],1,1])
    antenna_mesh_x = np.tile(np.expand_dims(antenna_position_x, (2,1)), [1, azi_mesh.shape[0], azi_mesh.shape[1]])
    # print(f"azi_mesh_vector shape: {azi_mesh_vector.shape}" )
    # print(f"antenna_mesh_x shape: {antenna_mesh_x.shape}" )
    

    ele_mesh_vector = np.tile(ele_mesh, [params["n_channels"],1,1])
    antenna_mesh_y = np.tile(np.expand_dims(antenna_position_y, (2,1)), [1, ele_mesh.shape[0], ele_mesh.shape[1]])


    beam_vector_x = np.exp(1j*2*np.pi*np.sin(azi_mesh_vector/180*np.pi)*np.cos(ele_mesh_vector/180*np.pi)*antenna_mesh_x)
    beam_vector_y = np.exp(1j*2*np.pi*np.sin(ele_mesh_vector/180*np.pi)*antenna_mesh_y)


    # print(f"beam_vector_x shape: {beam_vector_x.shape}" )
    # print(f"beam_vector_x angles: {np.array2string(np.angle(beam_vector_x)*(180/np.pi),precision=2,suppress_small=True)}" )



    beam_vector = beam_vector_x*beam_vector_y

    # print(f"beam_vector shape: {beam_vector.shape}" )
    # print(f"beam_vector angles: {np.array2string(np.angle(beam_vector)*(180/np.pi),precision=2,suppress_small=True)}" )

    beam_vector_flat = beam_vector.reshape([params["n_channels"], len(ele_mesh_range) * len(azi_mesh_range)])



    # the use:
    #steering_spectrum_course = np.abs(np.matmul(data['data_bin'][i,:]/self.bm0*self.win_full, self.beam_vector_reduced_flat).reshape([len(ele_mesh_range), len(azi_mesh_range)]))
    output = {"beam_vector_flat": beam_vector_flat, "azi_mesh_range": azi_mesh_range, "ele_mesh_range": ele_mesh_range, "calib":params["doa_offset_calibration_array"] }
    return output


DoA_dict_precalc = create_DFT_matrix(params)



def doDoA_CustomMatrixDFT():
    print("Implement this")
    pass

def doDoA_ZeroPadFFT(data, axis):
    params["azi_resolution_N"]
    params["ele_resolution_N"]
    print("Implement this")
    # np.fft.fft2(,axes=[])

    pass


if __name__ == "__main__":
    create_DFT_matrix(params)
    # sliders = {"Frame": 10,"Range":10}
    # out = process(None,sliders)
    # print(out)
    pass
