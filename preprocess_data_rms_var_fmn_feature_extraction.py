#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import scipy
from scipy.signal import iirnotch, butter, filtfilt, lfilter



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
        threshold_array = np.mean(np.abs(temp_epoch),axis=1).reshape(4,1) * multiplier

        #determine if any value in a channel is greater than its threshold 
        result = np.any(np.abs(temp_epoch) > threshold_array, axis=1)

        #if all values in each channel are less than each channel's threshold, save the epoch to an
        #output matrix called out_data
        if all(result) == False:
            out_data.append(temp_epoch)
    #transpose the out_data to match the input array dimensions 
    out_data = np.transpose(out_data,(1,2,0))
    return out_data

def get_var_trial(data):
    ch_var = []
    for ch  in range(len(data)):
        ch_data = data[ch,:,:]
        var_list = []
        for ep in range(ch_data.shape[1]):
            epoch_data = ch_data[:,ep]
            epoch_var = epoch_data.var()
            var_list.append(epoch_var)    
        ch_var.append(var_list)
    var_pd = pd.DataFrame(ch_var)
    mean_var = var_pd.mean(axis=1)
    return mean_var


def get_rms_trial(data):
    ch_rms = []
    for ch in range(len(data)):
        ch_data = data[ch,:,:]
        rms_list = []
        for ep in range(ch_data.shape[1]):
            epoch_data = ch_data[:,ep]
            epoch_rms = np.sqrt(np.mean(epoch_data**2))
            rms_list.append(epoch_rms)
        ch_rms.append(rms_list)
    rms_pd = pd.DataFrame(ch_rms)
    mean_rms = rms_pd.mean(axis=1)
    return mean_rms


def get_fmn_trial(f, psd):
    fmn_trial = []
    fs = 250
    for ch in range(len(psd)):
        ch_data = psd[ch,:]
        fmn = sum(f*ch_data)/sum(ch_data)
        fmn_trial.append(fmn)
    return fmn_trial

#input file path 
path = '/Users/sonali/Desktop/NEL_PROJECT/nov7/'

#create a list of .txt. files in the data folder
trial_list = [trials for trials in os.listdir(path) if '.txt' in trials]

variances = []
rms = []
fmn = []
#sort through all trials 
for val,act in enumerate(trial_list):
        #import .txt file as a pandas dataframe
        data=pd.read_csv(os.path.join(path,act), header=0,sep=',')
        #set data frequency
        fs = 250
        
        #set time range
        t = np.arange(1/fs, len(data)/fs + 1/fs,1/fs)

        #create a list of all channels
        channel_list = [' EXG Channel 0',' EXG Channel 1',' EXG Channel 2',' EXG Channel 3']

        
        all_channels = []
        psd = []
        for channel in channel_list:
            #sort through all channels, trim the epoch data, and append to a 3D matrix called all_channels
            (f, S)= scipy.signal.welch(data[channel].values, fs, nperseg=1024)
            psd.append(S)
            
            all_channels.append(epoch_data(10.08, preprocess_data(data[channel].values,fs), t, 1.5))

        #extract outliers from the 3D matrix where the first axis represents channels, the second axis
        #represents is time, and third axis is the epoch number
        #trimmed_data is the data matrix with the outlier epochs excluded
        psd = np.array(psd)
        
        
        trimmed_data = extract_outlier_epochs(all_channels,8)
        
        trial_variance = get_var_trial(trimmed_data)
        variances.append(trial_variance)
        
        trial_rms = get_rms_trial(trimmed_data)
        rms.append(trial_rms)

        trial_fmn = get_fmn_trial(f, psd)
        fmn.append(trial_fmn)





