import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.signal import argrelextrema
from scipy.optimize import root, leastsq

import sys

import h5py

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

def shift(array, scan_area):
    DC_even = []
    SYM_even= []
    DC_odd = []
    SYM_odd = []
    DC_all = []
    SYM_all = []
    shifting_range = array.shape[0]-scan_area
    
    for i in range(shifting_range):
        window = array[i:i+scan_area,:]
        rowsum = np.sum(window, axis = 1, dtype=np.float64)
        DC, SYM = DC_and_SYM(rowsum)
        DC_all.append(DC)
        SYM_all.append(SYM)
        DC_even.append(np.real(DC))
        SYM_even.append(np.real(SYM))
        DC_odd.append(np.imag(DC))
        SYM_odd.append(np.imag(SYM))
    return np.asarray(DC_all), np.asarray(SYM_all)

def cross_correlate(array, test_function = 'odd'):
    if test_function == 'odd':
        test = np.linspace(-1,1, array.shape[1])
    corr = []
    for i in range(array.shape[1]):
        corr.append(np.correlate(array, test))
    data_min = argrelextrema(np.asarray(corr), np.less)
    return data_min

def plot_DC_and_SYM(DC_all, SYM_all):
    plt.figure(figsize=(10,10))
    
    plt.subplot(321)
    plt.plot(np.abs(DC_all))
    plt.title('Absolute value of DC freq')
    plt.subplot(322)
    plt.plot(np.abs(SYM_all))
    plt.title('Absolute value of SYM freq')
    
    plt.subplot(323)
    plt.title('Even spectrum of DC freqency')
    plt.plot(np.real(DC_all))
    plt.subplot(324)
    plt.title('Odd sepctrum of DC frequency')
    plt.plot(np.imag(DC_all))
    
    plt.subplot(325)
    plt.title('Even spectrum of SYM frequency')
    plt.plot(np.real(SYM_all))
    plt.subplot(326)
    plt.title('Odd spectrum of SYM frequency')
    plt.plot(np.imag(SYM_all))

    plt.show()

    
def SYM_odd_root_poly(SYM_odd, scan_area, degree):
    # fit data to polynomial
    x = len(SYM_odd)
    N = np.arange(x)
    Ep = np.polyfit(N,SYM_odd,15)
    fitted = np.poly1d(Ep)
    root = np.real([a for a in np.roots(fitted.coef) if np.imag(a) == 0 and 0.<=np.real(a)<=x])[::-1]
    return root+scan_area/2

def SYM_odd_data_min(SYM_odd, scan_area):
    # find the minimum (zero) of data set (not polynomial)
    data_min = argrelextrema(abs(np.asarray(SYM_odd)), np.less)[0]
    return data_min+scan_area/2