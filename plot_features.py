## MAIN FILE FOR PROJECT ##
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from scipy.signal import iirnotch, butter, filtfilt, lfilter
from preprocess_data_with_epoch_extraction import extract_outlier_epochs, import_data 
from preprocess_data import epoch_data,preprocess_data
from extract_features import calc_features,extract_features_trials,get_var_trial,get_fmn_trial,get_rms_trial
from matplotlib.colors import LinearSegmentedColormap


path = '/Users/zeynepozkaya/Desktop/CMU FALL 23/Neural Engineering Lab/Final Project/drive-download-20231114T015003Z-001/'
fs = 250 
trimmed_data, psd = import_data(path, fs)
t = np.arange(1/fs, len(trimmed_data[1][0:1, :, 1])/fs + 1/fs,1/fs)
data = pd.read_csv('hand_fist.csv')
t_2 = np.arange(1/fs, len(data)/fs + 1/fs,1/fs)


'''
Script for plotting epoching example 
'''
def make_epoch_plot():
    test = epoch_data(10.08, preprocess_data(data.EXGChannel2,fs), t_2, 1.5)
    t = np.arange(1/fs, len(test)/fs + 1/fs,1/fs)

    fig = plt.figure(figsize=(8, 4))

    # Define the grid: 3 rows, 2 columns
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)  # Top plot spanning both columns
    ax2 = plt.subplot2grid((3, 2), (1, 0))  # Bottom-left plot
    ax3 = plt.subplot2grid((3, 2), (1, 1))  # Bottom-right plot

    ax1.plot(t_2, preprocess_data(data.EXGChannel2,fs))
    ax1.set_xlim(10,15)
    ax2.plot(t,test[:,1],color='red')
    ax3.plot(t,test[:,2],color='red')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax1.set_xlabel("Time [sec]")
    ax2.set_xlabel("Time [sec]")
    ax3.set_xlabel("Time [sec]")

    ax1.set_ylabel("EMG  [mV]")
    ax2.set_ylabel("EMG  [mV]")
    ax3.set_ylabel("EMG  [mV]")

    ax2.set_title("Epoch 1")
    ax3.set_title("Epoch 2")

    ax1.set_title("Hand Fist - Channel 2",fontweight='bold')

    ax1.set_ylim(-500, 500)      
    ax2.set_ylim(-500, 500)      
    ax3.set_ylim(-500, 500)      

    #plt.show()



"""
Bar plot plotting script for features based on channel or channel average - note 
still needs error bars for final report 

INPUT:
iemg, mav, ssi, fms, rms, mn 
avg_chan: specify if want to plot average of channels or individual channels, 
avg_epoch: specify if want to plot average of channels or individual epochs, 

OUTPUT:
none
"""
def plot_features(iemg, mav, ssi, fmd, fmn, var, rms, avg_chan, avg_epoch):
    values = [iemg, mav, ssi, fmd,fmn, var, rms]
    chan_labels = ['Chan 1', 'Chan 2', 'Chan 3', 'Chan 4']
    titles = ['hand_fist', 'index_finger_point', 'wrist_up', 'wrist_down', 'two_finger_pinch',
                'wrist_right', 'wrist_left', 'hand_open']
    labels = ['IEMG', 'MAV', 'SSI', 'FMD', 'FMN','RMS','VAR']
    
    if  avg_epoch and avg_chan:
        # Create subplots in a 2x4 layout
        fig, axs = plt.subplots(2, 4, figsize=(12, 6))

        # Flatten the axs array for easy iteration
        axs = axs.flatten()

        # Define a custom green-blue color map
        colors = [(0, 0.5, 0), (0, 0.7, 0.7), (0, 0.9, 1), (0, 0.7, 0.7), (0, 0.5, 0)]
        n_bins = 8
        cmap_name = "green_blue_wave"
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


        for i in range(7): # change to range 7 once other features integrated 
            axs[i].bar(["1", "2","3", "4","5", "6","7", "8",],values[i],color=custom_cmap(np.linspace(0, 1, len(titles))))
            axs[i].set_title(labels[i])
            #axs[i].set_xticklabels(titles, rotation='vertical')

        axs[7].axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=custom_cmap(x)) for x in np.linspace(0, 1, len(titles))]
        fig.legend(handles, titles, loc='lower right',bbox_to_anchor=(.9, 0.1),title='Trial Gestures', facecolor='white')


        # Adjust layout for better spacing
        plt.tight_layout()
        plt.suptitle('Epoch and Channel Averaged Feature Extraction',fontweight='bold')



        # Display the plot
        plt.show()
    
    if avg_epoch and not avg_chan:
        i = 2
        colors = [(0, 0, 0.5), (0, 0.3, 0.7), (0, 0.6, 1), (0, 0.3, 0.7)]
        n_bins = 4
        cmap_name = "blue_color_wave"
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


        fig, axs = plt.subplots(2,4, figsize=(12, 6))
        axs = axs.flatten()

        print(np.shape(values[1][i]))
        for k in range(7): # change to range 7 once other features integrated 
            axs[k].bar(chan_labels, values[k][i],color=custom_cmap(np.linspace(0, 1, len(labels))))
            axs[k].set_title(labels[k])

        axs[7].axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=custom_cmap(x)) for x in np.linspace(0, 1, len(titles))]
        fig.legend(handles, chan_labels, loc='lower right',bbox_to_anchor=(.9, 0.1),title='Channels', facecolor='white')


        plt.tight_layout()
        plt.suptitle("Index Finger Point",fontweight='bold')
        plt.show()






iemg, mav, ssi, fmd, fmn, var, rms = extract_features_trials(trimmed_data,psd, False, True)
plot_features(iemg, mav, ssi, fmd, fmn, var, rms, False,True)
            
