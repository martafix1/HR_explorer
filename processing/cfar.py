
import numpy as np
import scipy



def cell_average_CFAR(data,guard_n,train_n):
    treshold =  np.zeros_like(data)

    kernel_size = train_n + guard_n + 1 + guard_n + train_n

    kernel = np.zeros(kernel_size)

    #norm_w = (train_n*2.0)/kernel_size wrong
    norm_w = 1/(train_n*2)
    
    kernel[0:(train_n)] = norm_w
    kernel[-(train_n):] = norm_w
    # print(kernel)
    treshold =  np.convolve(data,kernel,mode="same")

    return treshold

def cell_average_CFAR_2D(data: np.ndarray ,guard_0,guard_1,train_0,train_1,dim0_i,dim1_i):

    kernel_n_0 = train_0 + guard_0 + 1 + guard_0 + train_0
    kernel_n_1 = train_1 + guard_1 + 1 + guard_1 + train_1

    kernel = np.ones((kernel_n_0,kernel_n_1))

    kernel[train_0:-train_0,
           train_1:-train_1] = 0
    
    kernel /= kernel.sum() # normalize
    # print(kernel)

    full_kernel_shape = [1] * data.ndim
    full_kernel_shape[dim0_i] = kernel_n_0
    full_kernel_shape[dim1_i] = kernel_n_1

    full_kernel = kernel.reshape(full_kernel_shape)

    threshold = scipy.signal.convolve2d(data, full_kernel, mode='same')

    return threshold


def cell_max_CFAR_2D(data: np.ndarray ,guard_0,guard_1,train_0,train_1):

    kernel_n_0 = train_0 + guard_0 + 1 + guard_0 + train_0
    kernel_n_1 = train_1 + guard_1 + 1 + guard_1 + train_1

    kernel = np.ones((kernel_n_0,kernel_n_1))

    s0 = slice(train_0, -train_0 if train_0 != 0 else None)
    s1 = slice(train_1, -train_1 if train_1 != 0 else None)

    kernel[s0,
           s1] = 0
    
    # kernel 
    # print(kernel)

    footprint = kernel.astype(bool)
    # print(footprint)

    threshold = scipy.ndimage.maximum_filter(data, footprint=footprint, mode='constant', cval=-np.inf)

    

    return threshold



if __name__ == "__main__":

    cell_average_CFAR(0,1,2)


    fakeData = np.ones((5,10))
    cell_average_CFAR_2D(fakeData,1,1,1,1,0,1)




