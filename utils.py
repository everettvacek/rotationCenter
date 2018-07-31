import numpy as np

def snr_calc(array, noise, exposure=None):
    if exposure == None:
        snr = np.mean(array)/np.std(noise-array)
    else:
        snr = np.mean(array/np.max(array)*exposure)/np.std(noise-array/np.max(array)*exposure)
    return snr

def add_noise(array, exposure = None, bit_depth = None):
    if exposure == None:
        noise = array #np.random.poisson(array/np.max(array)*exposure)
        snr = snr_calc(array, noise, exposure)
    else:
        noise = np.random.poisson(array/np.max(array)*exposure)
        snr = snr_calc(array, noise, exposure)
    if bit_depth == None:
        pass
    if bit_depth == 8:
        noise = np.array(np.floor(noise)).astype(dtype=np.uint8)
        #noise = np.array(np.floor(noise*(2**8-2)/np.max(noise))).astype(dtype=np.uint8)
    if bit_depth == 16:
        noise = np.array(np.floor(noise*(2**16-2)/np.max(noise))).astype(dtype=np.uint16)
    return noise, snr

def DC_and_SYM(array):
    #input is 1d array. output is the even and odd parts of the DC and SYM spectrum
    array_fft = np.fft.fft(array)
    DC = array_fft[0]
    SYM = array_fft[1]
    return DC, SYM

