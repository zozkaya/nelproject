import numpy as np 
import pandas as pd
import scipy as scipy
from scipy.signal import iirnotch, butter, filtfilt, lfilter
import matplotlib.pyplot as plt


def get_var_trial(data):
    ch_var = []
    for ch  in range(len(data)):
        ch_data = data[ch,:,:]
        var_list = []
        for ep in range(ch_data.shape[1]):
            epoch_data = ch_data[:,ep]
            epoch_var = data.var()
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

"""
calc_time_features: calculates the time domain features (iemg, mav, and ssi) for given data

INPUTS:
trimmed_data: a 3D matrix, where channels (1-4) is the first axis, the time is the second axis, and the third axis is the epoch number
window_size: the size of rolling windows in unit of sample #

OUTPUT:
avg_iemg, avg_mav, avg_ssi,avg_fms: each output, time features (iemg, mav, and ssi), is a 2D matrix where first axis is channel # and second axis is epoch # 
"""
def calc_features(data,psd, window_size=374):
    # Store the dimentions into variables
    num_ch, num_samples, num_epochs = data.shape
    num_windows = num_samples - window_size + 1

    # Declare outputs
    avg_iemg = np.zeros((num_ch, num_epochs))
    avg_mav = np.zeros((num_ch, num_epochs))
    avg_ssi = np.zeros((num_ch, num_epochs))
    avg_fmd = np.zeros((num_ch, num_epochs))
    avg_var = np.zeros((num_ch, num_epochs))
    avg_rms = np.zeros((num_ch, num_epochs))
    avg_fmn = np.zeros((num_ch, num_epochs))

    for channel in range(num_ch):
        for epoch in range(num_epochs):
            total_iemg = 0
            total_mav = 0
            total_ssi = 0
            total_fmd = 0
            total_var = 0
            total_rms = 0 
            total_fms = 0

            for window_str in range(num_windows):
                window_end = window_str + window_size
                window_data = data[channel, window_str : window_end, epoch]

                abs_window_data = np.abs(window_data)
                sq_window_data = np.square(window_data)

                iemg = np.sum(abs_window_data)

                total_iemg += iemg
                total_mav += iemg / window_size
                total_ssi += np.sum(sq_window_data)
                total_fmd += np.mean(.5*sum(scipy.signal.welch(window_data, 250, nperseg=64)))
                total_var += window_data.var()
                total_rms += np.sqrt(np.mean(window_data**2))
                total_fms += sum(250*scipy.signal.welch(window_data, 250, nperseg=64))/sum(scipy.signal.welch(window_data, 250, nperseg=64))
            
            avg_iemg[channel, epoch] = total_iemg / num_windows
            avg_mav[channel, epoch] = total_mav / num_windows
            avg_ssi[channel, epoch] = total_ssi / num_windows
            avg_fmd[channel, epoch] = total_fmd/ num_windows
            avg_var[channel, epoch] = total_var/ num_windows
            avg_rms[channel, epoch] = total_rms/ num_windows
            avg_fmn[channel, epoch] = total_rms/ num_windows

    return avg_iemg, avg_mav, avg_ssi, avg_fmd, avg_var, avg_rms, avg_fmn


def real_time_calc_features(window_data_all):

    total_iemg = []
    total_mav = []
    total_ssi = []
    total_fmd = []
    total_var = []
    total_rms = []
    total_fmn = []




    for i in range((len(window_data_all[1]))): #window_data[1] is channel size
        window_size = len(window_data_all[0])
        window_data = window_data_all[:,i]

        abs_window_data = np.abs(window_data)
        sq_window_data = np.square(window_data)

        iemg = np.sum(abs_window_data)

        total_iemg.append(iemg)
        total_mav.append(iemg / window_size)
        total_ssi.append(np.sum(sq_window_data))
        total_fmd.append(np.mean(.5*sum(scipy.signal.welch(window_data, 250, nperseg=64))))
        total_var.append(window_data.var())
        total_rms.append(np.sqrt(np.mean(window_data**2)))
        total_fmn.append(np.sum(250*scipy.signal.welch(window_data, 250, nperseg=64))/np.sum(scipy.signal.welch(window_data, 250, nperseg=64)))

    combined_features = np.hstack([(total_iemg),(total_mav),(total_ssi),(total_fmd),(total_fmn),(total_var),(total_rms)])

    return combined_features


"""
Extracts features for each trial uses calc_features function and 
returns 8 element array for each feature 

INPUTS:
trimmed_data: data for single file 
avg_chan: if True average across channels 
avg_epoch: if False average across epochs 

RETURNS:
iemg_all: extracted iemg vector for all files (8x4xepoch_len)
    - iemg_all[0] - hand_fist.txt
    - iemg_all[1] - index_finger_point.txt
    - iemg_all[2] - wrist_up.txt
    - iemg_all[3] - wrist_down.txt
    - iemg_all[4] - two_finger_pinch.txt
    - iemg_all[5] - wrist_right.txt
    - iemg_all[6] - wrist_left.txt
    - iemg_all[7] - hand_open.txt
mav_all: extracted mav vector for all files, same format as iemg_all (8x4xepoch_len)
ssi_all :extracted ssi vector for all files, same format as iemg_all (8x4xepoch_len)
fmd_all: extracted fmd vector for all files, same format as iemg_all (8x4xepoch_len)

if averaging flags are set, returns data with that dimension averaged 
"""
def extract_features_trials(trimmed_data, psd, avg_chan, avg_epoch):

    iemg_all = []
    mav_all = []
    ssi_all = []
    fmd_all = []
    var_all = []
    rms_all = []
    fmn_all = []

    for i in range(8):
        iemg, mav, ssi, fmd,var, rms, fmn = calc_features(trimmed_data[i],psd)
        if (avg_chan and not avg_epoch):
            iemg_all.append(np.mean(iemg, axis=0).T)
            mav_all.append(np.mean(mav, axis=0).T)
            ssi_all.append(np.mean(ssi, axis=0).T)
            fmd_all.append(np.mean(fmd, axis=0).T)
            var_all.append(np.mean(var, axis=0).T)
            rms_all.append(np.mean(rms, axis=0).T)
            fmn_all.append(np.mean(fmn, axis=0).T)
        elif (avg_epoch and not avg_chan):
            iemg_all.append(np.mean(iemg, axis=1).T)
            mav_all.append(np.mean(mav, axis=1).T)
            ssi_all.append(np.mean(ssi, axis=1).T)
            fmd_all.append(np.mean(fmd, axis=1).T)
            var_all.append(np.mean(var, axis=1).T)
            rms_all.append(np.mean(rms, axis=1).T)
            fmn_all.append(np.mean(fmn, axis=1).T)
        elif (avg_epoch and avg_chan):
            iemg_all.append(np.mean(np.mean(iemg, axis=0)).T)
            mav_all.append(np.mean(np.mean(mav, axis=0)).T)
            ssi_all.append(np.mean(np.mean(ssi, axis=0)).T)
            fmd_all.append(np.mean(np.mean(fmd, axis=0)).T)
            var_all.append(np.mean(np.mean(var, axis=0)).T)
            rms_all.append(np.mean(np.mean(rms, axis=0)).T)
            fmn_all.append(np.mean(np.mean(fmn, axis=0)).T)
        else:
            iemg_all.append(iemg.T)
            mav_all.append(mav.T)
            ssi_all.append(ssi.T)
            fmd_all.append(fmd.T)
            fmn_all.append(fmn.T)
            var_all.append(var.T)
            rms_all.append(rms.T)


        
    
    return iemg_all, mav_all, ssi_all, fmd_all, fmn_all, var_all, rms_all 

