import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import scipy as scipy 
from scipy.signal import iirnotch, butter, filtfilt, lfilter
from find_MVC import calculate_MVC

## GLOBALS 
# input file path 
path = '/Users/laurenparola/Desktop/NEL_FinalProject/nov7_prelim/'
path_mvc = ''
start_time = 7.5 # number of seconds before starting trial based on bpm of metronome 
chan_num = 8 # specify channel number for processing 

# set data frequency
fs = 250

#WRITTEN ASSUMING THIS ORDER ?? SHOULD IT BE ALPHABETIC?
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
def epoch_data(start_time, data, t,epoch_len,mvc):

    idx = np.where(t >= epoch_len)[0][0] #index corresponding to epoch length 
    idx1 = np.where(t >= start_time)[0][0] #index of the start time

    epoched_data = []; 
    rest_state = []; 


    # epoch the data starting at first epoch and looking at 
    # every other "epoch_len" second chunk 
    for val, i in enumerate(np.arange(idx1, len(data) - idx,idx)):
        epoch = data[i:i+idx]
        t_epoch = t[i:i+idx]
        if (val % 2 == 0):
            epoched_data.append(epoch/mvc) # normalize to mvc 
        elif (val % 2 != 0):
            rest_state.append(epoch/mvc) # normalize to mvc 

    epoched_data = np.array(epoched_data).T  # transpose for the desired format
    rest_state = np.array(rest_state).T
    
    return epoched_data, rest_state 

def extract_outlier_epochs(all_trial_array,multiplier):
    #inputs:
        #all_trial_array: a 3D matrix, where channels (1-4) is the first axis, the time is the second
        #axis, and the third axis is the epoch number
        #multiplier: multiplier used for the threshold, the threshold value is used as any values
        #greater than the mean of the absolute value of the data greater than some multiplier.

    #returns:
        #out_data: an array that with epochs removed that contain data above a certain threshold
    
    out_data = []

    for epoch in range(np.shape(np.array(all_trial_array))[2]):
        #extract a specific epoch
        temp_epoch = np.array(all_trial_array)[:,:,epoch]

        #create a 4 by 1 array of threshold based on the average value of each channels times a certain
        #multiplier
        threshold_array = np.mean(np.abs(temp_epoch),axis=1).reshape(8,1) * multiplier
        
        #determine if any value in a channel is greater than its threshold 
        result = np.any(np.abs(temp_epoch) > threshold_array, axis=1)

        #if all values in each channel are less than each channel's threshold, save the epoch to an
        #output matrix called out_data
        if all(result) == False:
            out_data.append(temp_epoch)
    #transpose the out_data to match the input array dimensions 
    out_data = np.transpose(out_data,(1,2,0))
    
    return out_data

"""
INPUTS: 
path
fs 
chan_used: 4 if you want to use 4 channels, 8 if you want to use all 8 
chan_num: specify number of channels used 
"""
def import_data(path,fs,chan_num,chan_used,mvc_dict):
    #create a list of .txt. files in the data folder
    trial_list = [trials for trials in os.listdir(path) if '.txt' in trials]
    extracted_trials = []
    rest_trials = []
    #sort through all trials

    for val,act in enumerate(trial_list):
        i = 0 
        #import .txt file as a pandas dataframe
        data=  pd.read_csv(os.path.join(path,act), header=0,sep=',',skiprows=4)
        
        #set time range
        t = np.arange(1/fs, len(data)/fs + 1/fs,1/fs)

        #create a list of all channels
        if (chan_num == 4):
            channel_list = [' EXG Channel 0',' EXG Channel 1',' EXG Channel 2',' EXG Channel 3']
        elif (chan_num == 8 and chan_used == 8):
            channel_list = [' EXG Channel 0',' EXG Channel 1',' EXG Channel 2',' EXG Channel 3',
                            ' EXG Channel 4',' EXG Channel 5',' EXG Channel 6',' EXG Channel 7']
        elif (chan_num == 8 and chan_used == 4):
            channel_list = [' EXG Channel 0',' EXG Channel 2',' EXG Channel 4',' EXG Channel 6'] #take every other channel

      #  start_time = start_times_vec[i]
        key = act[0:-4]+"_MVC"
        mvc_mat = mvc_dict[key]

        active_channels = []
        rest_channels = []
        for channel in channel_list:
            #sort through all channels, trim the epoch data, and append to a 3D matrix called all_channels
            if (chan_num == 8 and chan_used == 4):
                mvc = mvc_mat[i*2]
            else:
                mvc = mvc_mat[i]
            i += 1 

            active_epochs, rest_epochs = epoch_data(start_time, preprocess_data(data[channel].values,fs), t, 1.5,mvc)
            active_channels.append(active_epochs)
            rest_channels.append(rest_epochs)

        #extract outliers from the 3D matrix where the first axis represents channels, the second axis
        #represents is time, and third axis is the epoch number
        #trimmed_data is the data matrix with the outlier epochs excluded
        trimmed_data = extract_outlier_epochs(active_channels,8,)
        rest_data = extract_outlier_epochs(rest_channels,8,)
        extracted_trials.append(trimmed_data)
        rest_trials.append(rest_data)

        
        psd = []
        for channel in channel_list:
            #sort through all channels, trim the epoch data, and append to a 3D matrix called all_channels
            (f, S)= scipy.signal.welch(data[channel].values, fs, nperseg=1024)
            psd.append(S)

        psd = np.array(psd)
    trial_order = [trial.replace('.txt','') for trial in trial_list]
    return extracted_trials, rest_trials, psd, trial_order
            
        

        
