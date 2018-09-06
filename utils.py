import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.signal import argrelextrema, cwt, ricker
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
    DC = array_fft[3]
    SYM = 0
    for i in range(1):
        SYM += array_fft[2*i+1]
    return DC, SYM

def shift(array, scan_area, plot = None, shifting_range = 'Full'):
    # if plot = # then on plot per # shifts will be printed
    DC_even = []
    SYM_even= []
    DC_odd = []
    SYM_odd = []
    DC_all = []
    SYM_all = []
    SYM_min = []
    if shifting_range == 'Full':
        shifting_range = array.shape[0]-scan_area
    else:
        shifting_range = shifting_range
    
    for i in range(shifting_range):
        window = array[i:i+scan_area,:]
        rowsum = np.sum(window, axis = 1, dtype=np.float64)
        ## Collect spectrum data
        DC, SYM = DC_and_SYM(rowsum)
        DC_all.append(DC)
        SYM_all.append(SYM)
        DC_even.append(np.real(DC))
        SYM_even.append(np.real(SYM))
        DC_odd.append(np.imag(DC))
        SYM_odd.append(np.imag(SYM))
        ## produce spectrum
        DC_spec = np.zeros(rowsum.shape, dtype = 'complex')
        SYM_spec = np.zeros(rowsum.shape, dtype = 'complex')
        DC_spec[1] += DC
        SYM_spec[1] += SYM #+ DC
        #SYM_spec[rowsum.shape[0]-1] += SYM
        iDC_spec = np.fft.ifft(DC_spec)
        iSYM_spec = np.fft.ifft(SYM_spec)
        iSYM_even = np.fft.ifft(np.real(SYM_spec))
        iSYM_odd = np.fft.ifft(np.imag(SYM_spec)*1j) #+ np.real(np.fft.ifft(np.imag(SYM_spec)*1j))
        iDC_odd = np.fft.ifft(np.imag(DC_spec)*1j)
        #SYM_min.append(np.trapz(np.abs(iSYM_odd)))
        try:
            SYM_min.append(argrelextrema(iSYM_spec, np.less)[0][0] + i)
        except:
            pass
        if plot != None and len(plot) > 1 and i in plot:
            plt.figure(figsize=(12,5))
            plt.subplot(121)
            plt.imshow(array[i:i+scan_area,:])
            plt.subplot(122)
            plt.plot((rowsum-np.min(rowsum))/np.max(rowsum-np.min(rowsum))**.5, label = 'rowsum')
            plt.plot((iSYM_spec-np.min(iSYM_spec))/np.max(iSYM_spec-np.min(iSYM_spec))**.5, label = 'iSYM')
            plt.plot((iSYM_even-np.min(iSYM_even))/np.max(iSYM_even-np.min(iSYM_even))**.5, label = 'iSYM_even')
            plt.plot((iSYM_odd-np.min(iSYM_odd))/np.max(iSYM_odd-np.min(iSYM_odd))**.5, label = 'iSYM_odd')
            plt.title('Window Center = '+ str(i+scan_area//2))#+ ', SYM center = '+ str(argrelextrema(iSYM_spec, np.less)[0][0] + i) + '    '+ str(i))
            plt.legend()
            plt.show()
        
        elif plot != None and len(plot)==1 and i % plot[0] == 0:
            plt.figure(figsize=(12,5))
            plt.subplot(121)
            plt.imshow(array[i:i+scan_area,:])
            plt.subplot(122)
            plt.plot((rowsum-np.min(rowsum))/np.max(rowsum-np.min(rowsum))**.5, label = 'rowsum')
            plt.plot((iSYM_spec-np.min(iSYM_spec))/np.max(iSYM_spec-np.min(iSYM_spec))**.5, label = 'iSYM')
            plt.plot((iSYM_even-np.min(iSYM_even))/np.max(iSYM_even-np.min(iSYM_even))**.5, label = 'iSYM_even')
            plt.plot((iSYM_odd-np.min(iSYM_odd))/np.max(iSYM_odd-np.min(iSYM_odd))**.5, label = 'iSYM_odd')
            plt.title('Window Center = '+ str(i+scan_area//2))#+ ', SYM center = '+ str(argrelextrema(iSYM_spec, np.less)[0][0] + i) + '    '+ str(i))
            plt.legend()
            plt.show()
        
    plt.plot(np.asarray(SYM_min))
    plt.title('Mode of the minimum of Even component = ' + str(scipy.stats.mode(SYM_min)[0][0]))
    plt.show()
    
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
    Ep = np.polyfit(N,SYM_odd,degree)
    fitted = np.poly1d(Ep)
    root = np.real([a for a in np.roots(fitted.coef) if np.imag(a) == 0 and 0.<=np.real(a)<=x])[::-1]
    return root+scan_area/2, fitted

def SYM_odd_data_min(SYM_odd, scan_area):
    # find the minimum (zero) of data set (not polynomial)
    data_min = argrelextrema(abs(np.asarray(SYM_odd)), np.less)[0]
    return data_min+scan_area/2

def fit_to_cos(array):
    N = len(array)
    data = array
    t = np.linspace(0, 4*np.pi, N)
    guess_mean = np.mean(data)
    guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
    guess_phase = 0
    guess_freq = 1
    guess_amp = np.max(data)-np.abs(np.min(data))

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - data
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]
    
    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean
    
    return data_fit
