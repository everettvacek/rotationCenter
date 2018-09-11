import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import root, leastsq
from scipy.interpolate import UnivariateSpline

#from skimage import dtype_limits
from skimage.measure import shannon_entropy
from skimage.util import img_as_uint, img_as_ubyte

import h5py

import warnings

import math
def goertzel(samples, *freqs):
    """
    Adapted from author: Sebastien Piquemal https://github.com/sebpiq
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.
    """
    window_size = len(samples)
    f_step = 1
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = 1, 2
        k_start = f_start // f_step
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append(w_imag * d1)
    return results

def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

def tomobank_phantom_sino(path, make_360 = True, print_atters = False):
    f = h5py.File(path,'r')
    data = f['/exchange/data'][()]
    theta = f['/exchange/theta'][()]
    if make_360 and np.max(theta) <= 180:
        theta = np.hstack((theta,theta*2))
        data = np.concatenate((data, np.flip(data, axis = 2)), axis = 0)
    if print_atters:
        try:
            h_sino.visititems(print_attrs)
        except:
            print("CANNOT PRINT FILE TREE")
    f.close()
    
    return data, theta

def set_bit_depth(array, bit_depth):
    '''
    Set bit depth of image and scale values to prevent saturation or wrap around. 
    Most detectors use uint16.
    '''
    array = ((array/np.max(array))*2**bit_depth-1).astype(bit_depth)
    return array

def add_noise2(array, exposure = None, bit_depth = None):
    '''
    returnes the snr as peak snr for reduced exposure images, returns shannon entropy for exposure = None.
    '''
    ## Change bit_depth if necessary
    if bit_depth == 16:
        array = img_as_uint(array/np.max(array))
        clip = 2**16-1
    elif bit_depth == 8:
        noise = img_as_ubyte(array/np.max(array))
        clip = 2**8-1
    else:
        clip = None
    if exposure == None:
        noise = array
        snr = shannon_entropy(noise)
    else:
        noise = np.clip(np.random.poisson(array/np.max(array)*exposure), 0, clip)
        snr = shannon_entropy(noise)
    return noise, snr
'''
def contrast_ratio(array):
    ## Returns the ration of the difference of the highest and lowest pixel value
    ## to the highest and lowest possible pixel value for a given dtype
    dlimits = dtype_limits(array, clip_negative=False)
    limits = np.percentile(array, [1, 99])
    return (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
'''
def fourier_bin(array, fbins):
    ## Transforms 1D array and returns requested bins.
    ## array: 1D
    ## fbins: 1D array of bin requests (0:len(array))
    array_fft = np.fft.rfft(array)
    bins = []
    length = len(array)    
    for i in fbins:
        bins.append(array_fft[i])
    return bins

def window_bins(array, window_width, start_index, orientation = 'xt', fbins = 'all'):
    ## windows a sinogram and returns fourier bins
    ## orientation refers to which axis contains the x axis or the theta axis
    ## set bins to 1d array of desired bins.
    #if orientation == 'tx':
    #    array = np.transpose(array)
    window = array[start_index:start_index+window_width]
    #rowsum = np.sum(window, axis = 1, dtype=np.float64)
    
    #if fbins == 'all':
    #    bins = fourier_bin(rowsum, np.arange(len(rowsum)))
    
    #bins = goertzel(window, (1,2))
    bins = fourier_bin(window, fbins)
        
    return bins

def first_spectrum_min(array, window_width, orientation = 'xt'):
    spec_min = []
    for i in range(array.shape[0]-window_width):
        fft_spec = np.zeros(array.shape[0]-window_width, dtype='complex')
        fft_spec[1] = window_bins(array, window_width, i, fbins = [1])[0]
        spec = np.fft.ifft(fft_spec)
        #plt.plot(even_ifft)
        #plt.show()
        try:
            spec_min.append(argrelextrema(spec, np.less)[0][0] + i)
        except:
            pass
    return spec_min

def imimplot(array1, array2, plot, labels, title = '', scatter = False):
    plt.figure(figsize = (15,5))
    plt.subplot(131)
    plt.imshow(array1)
    plt.title(labels[0])
    plt.subplot(132)
    plt.imshow(array2)
    plt.title(labels[1])
    ax = plt.subplot(133)
    for i in range(len(plot)):
        if scatter == False:
            ax.plot(plot[i], label = labels[i])
            #ax.set_yscale('log')
        else:
            ax.scatter(plot[i], label = labels[i], linewidth = .1)
            #ax.set_yscale('log')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
def plot_odd_even(spectrum, titles = None):
    ## Plot odd and even components of a spectrum
    num = len(spectrum)
    plt.figure(figsize = (10,10))
    
    for i in range(num):
        plt.subplot(int(num*100+20+i+1))
        plt.plot(spectrum[i])
        if titles != None:
            plt.title(titles[i])
    plt.show()

def spline_root(array, window, smoothing = 3, s = None):
    ## Fit a 1D array to a polynomial. Returns both roots and fitted function
    if isinstance(array, complex):
        warnings.warn('WARNING: Input array is complex')
    spline = UnivariateSpline(np.arange(len(array)), array, k = smoothing, s = s)
    try:
        root = UnivariateSpline.roots(spline) + window/2 #np.real([a for a in np.roots(fitted.coef) if 0.<=np.real(a)<=N and np.imag(a)==0][::-1])
    except:
        root = [-1]
    
    return root, spline

def data_min(array, comparator=np.less):
    ## Find the extrema arguments of data set and return their x and y values
    ## If none are found returns [(-1,-1)]
    extrema = argrelextrema(array, comparator)[0]
    try:
        y = [array[a] for a in extrema]
        extrema_y = list(zip(extrema, y))
    except:
        extrema_y = [(-1,-1)]
    return extrema_y

def fit_to_cos2(array):
    N = len(array)
    data = array
    t = np.linspace(0, 4*np.pi, N)
    guess_mean = np.mean(data)
    guess_std = np.std(data)
    guess_phase = 0
    guess_freq = 1
    guess_amp = np.max(data)-np.abs(np.min(data))

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

    # Define the function to optimize, minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - data
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]
    
    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean
    
    return [est_amp, est_freq, est_phase, est_mean], fitted