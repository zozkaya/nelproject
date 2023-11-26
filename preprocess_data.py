import numpy as np 
import pandas as pd
from scipy.signal import iirnotch, butter, filtfilt, lfilter
import matplotlib.pyplot as plt



"""
Applies bandpass and notch filter to data 

INPUTS:
chan_data: timeseries for single channel 
fs: sampling frequency 

OUTPUTS:
data_filt = filtered data for a single channel 

"""
def preprocess_data (chan_data,fs):
    # Bandpass filter parameters

    nyquist = 0.5 * fs
    low = 5 / nyquist
    high = 90 / nyquist
    b, a = butter(4, [low, high], btype='band')
    data_filt = lfilter(b, a, chan_data)

    return data_filt

    

"""
The purpose of this function is to extract individual
trials from a large file of data.  

INPUTS:
start_time: time point for first trial 
data: single channel EMG recording 
t: corresponding time vector for data 
epoch_len: duration of each trial 

OUTPUT:
epoch_mat: 2D matrix where each row 
corresponds to a single epoch 


"""
def epoch_data(start_time, data, t,epoch_len):

    idx = np.where(t >= epoch_len)[0][0] #index corresponding to epoch length 
    idx1 = np.where(t >= start_time)[0][0] #index of the start time

    epoched_data = []; 


    # epoch the data starting at first epoch and looking at 
    # every other "epoch_len" second chunk 
    for i in range(idx1, len(data) - idx, idx * 2):
        epoch = data[i:i+idx]
        t_epoch = t[i:i+idx]
        epoched_data.append(epoch)

    epoched_data = np.array(epoched_data).T  # transpose for the desired format


    
    return epoched_data


