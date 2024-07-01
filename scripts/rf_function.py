import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import welch
from math import degrees, asin

def __cal_entropy(series):
    """ Calculate entropy of a given series """
    count_series = series.value_counts()
    probabilities = count_series / len(series)
    entropy = -sum(probabilities * np.log2(probabilities + 1e-8))  # Avoid log2(0)
    return entropy

def __fft_features(signal, Hz):
    # dominant frequency
    signal = signal.values
    signal = signal - np.mean(signal)  # Subtract mean
    n = len(signal)
    
    fft_signal = fft(signal)
    fft_mag = np.abs(fft_signal) / n
    freq = np.fft.fftfreq(n, d=1/Hz)  # Frequency with Hz parameter

    # Only consider positive frequencies
    positive_freqs = freq[np.where(freq >= 0)]
    positive_mags = fft_mag[np.where(freq >= 0)]
    
    # Dominant frequency and magnitude
    dominant_freq = positive_freqs[np.argmax(positive_mags)]
    dominant_mag = np.max(positive_mags)

    # Energy
    energy = sum(fft_mag**2) / n

    return energy, dominant_freq, dominant_mag

def __spectral_power(signal, Hz, band):
    # frequency domain features using Welch's method
    freqs, psd = welch(signal, fs=Hz, nperseg=len(signal))
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    power = np.sum(psd[band_mask])
    return power


def extract_acc_fet(data, Hz):

    # Ensure there are three axes x, y, z
    if len(data.columns) != 3:  
        return None

    feature_dict = {}

    # Vector Magnitude
    vm = np.sqrt((data ** 2).sum(axis=1))
    feature_dict['ACC_vm_min'] = vm.min()
    feature_dict['ACC_vm_max'] = vm.max()
    feature_dict['ACC_vm_std'] = vm.std()

    # Forward angle calculations
    magnitude = np.sqrt((data ** 2).sum(axis=1)).mean()
    x_axis_mean = data['x'].mean()
    z_axis_mean = data['z'].mean()
    feature_dict['ACC_ForBack_Angle'] = degrees(-asin(z_axis_mean / (magnitude + 0.00001))) # worth cancel

    for axis in ['x', 'z'] : # worth cancel y and z
        _, dominant_freq, _ = __fft_features(data[axis], Hz=Hz)  # Set Hz as needed
        feature_dict[f'ACC_dominantFr_{axis}'] = dominant_freq

    # Entropy
    feature_dict['ACC_entropy_x'] = __cal_entropy(data['x'])

    for axis in ['x']:
        power_0_25_1 = __spectral_power(data[axis], Hz, (0.25, 1.0))
        feature_dict[f'ACC_power_0_25_1_{axis}'] = power_0_25_1

    return pd.Series(feature_dict)