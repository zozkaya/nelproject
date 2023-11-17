import numpy as np 
import pandas as pd
from scipy.signal import iirnotch, butter, filtfilt, lfilter
import matplotlib.pyplot as plt


### Functions:

"""
txt_2_csv(): Makes .csv file from .txt file, skipping metadatas and sample datas

INPUTS:
in_file_path: a string of path to input file
out_file_path: a string of path to output file
"""
def txt_2_csv(in_file_path, out_file_path):
    with open(in_file_path, 'r') as file:
        contents = file.readlines()

    # Skipping metadata and starting from the data header line which starts from the 5th line
    data_lines = contents[4:]

    df_processed = pd.DataFrame([line.strip().split(',') for line in data_lines[1:]], columns=data_lines[0].strip().split(','))
    # Delete the spaces within each column name
    df_processed.columns = df_processed.columns.str.replace(' ', '')

    # Skipping sample data, starting from the 4th line
    df_processed = df_processed[3:]
    df_processed.to_csv(out_file_path, index=False)


"""
bandpass_filt(): Applies bandpass (low 5, high 90) to data, order default to 4

INPUTS:
chan_data: timeseries for single channel
fs: sampling frequency
low_cut: low cut for bandpass filter
high_cut: high cut for bandpass filter
order: default to order of 4

OUTPUTS:
data_filt = filtered data for a single channel 
"""
def bandpass_filt_ch(chan_data, fs, low_cut, high_cut, order=4):
    # Bandpass filter parameters
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order, [low, high], btype='band')
    data_filt = lfilter(b, a, chan_data)

    return data_filt


"""
filt_data(): Applies bandpass to data in channels 0-7, which is all the emg data

INPUTS:
raw_data: full raw data
header_list: list of strings of names of columns that want to be filtered (channel 0-7)
fs: sampling frequency
low_cut: low cut for bandpass filter
high_cut: high cut for bandpass filter

OUTPUTS:
data_filt = a DataFrame of filtered data of channels 0-7
"""
def filt_data(raw_data, header_list, fs, low_cut, high_cut):
    # Dictionary to store filtered data 
    filtered_data = {}

    # Add Sample index to dictionary
    filtered_data['SampleIndex'] = raw_data['SampleIndex']

    # Loop through each channel
    for header in header_list:
        # Filter current channel
        filtered_channel = bandpass_filt_ch(raw_data[header], fs, low_cut, high_cut)
        # Create a dictionary for each filtered channels
        filtered_data[header] = filtered_channel

    # Create a DataFrame with the filtered data
    filtered_df = pd.DataFrame(filtered_data)

    return filtered_df


"""
plot_time_series(): plot channels of emg recordings

INPUTS:
data: 4 channels of emg recordings
"""
def plot_time_series(data):
    plt.figure(num='Figure 1', figsize=(10, 6))  # Adjust the size of the plot as needed
    # plt.plot(filt_data.EXGChannel0, label='Filtered Data')
    for i in range(4):
        plt.plot(data[f'EXGChannel{i}'], label=f'Channel {i}')
    plt.title('Filtered Channels 0-3 Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Voltage (uV)')
    plt.xlim(0)
    plt.ylim(-750, 750)
    plt.legend()
    plt.show()


"""
epoch_data(): extracts individual trials from a large file of data

INPUTS:
start_time: time point for first trial 
data: single channel EMG recording 
t: corresponding time vector for data 
epoch_len: duration of each trial 

OUTPUT:
epoch_mat: 2D matrix where each row corresponds to a single epoch 
"""
def epoch_data(start_time, data, t, epoch_len):

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


"""
extract_outlier_epochs: remove epochs that contain data above threshold

INPUTS:
all_trial_array: a 3D matrix, where channels (1-4) is the first axis, the time is the second axis, and the third axis is the epoch number
multiplier: multiplier used for the threshold, the threshold value is used as any values greater than the mean of the absolute value of the data greater than some multiplier.
epoch_len: duration of each trial 

OUTPUT:
out_data: an array that with epochs removed that contain data above a certain threshold
"""
def extract_outlier_epochs(all_trial_array, multiplier):    
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


"""
calc_time_features: calculates the time domain features (iemg, mav, and ssi) for given data

INPUTS:
trimmed_data: a 3D matrix, where channels (1-4) is the first axis, the time is the second axis, and the third axis is the epoch number
window_size: the size of rolling windows in unit of sample #

OUTPUT:
avg_iemg, avg_mav, avg_ssi: each output, time features (iemg, mav, and ssi), is a 2D matrix where first axis is channel # and second axis is epoch # 
"""
def calc_time_features(data, window_size=5):
    # Store the dimentions into variables
    num_ch, num_samples, num_epochs = data.shape
    num_windows = num_samples - window_size + 1

    # Declare outputs
    avg_iemg = np.zeros((num_ch, num_epochs))
    avg_mav = np.zeros((num_ch, num_epochs))
    avg_ssi = np.zeros((num_ch, num_epochs))

    for channel in range(num_ch):
        for epoch in range(num_epochs):
            total_iemg = 0
            total_mav = 0
            total_ssi = 0

            for window_str in range(num_windows):
                window_end = window_str + window_size
                window_data = data[channel, window_str : window_end, epoch]

                abs_window_data = np.abs(window_data)
                sq_window_data = np.square(window_data)

                iemg = np.sum(abs_window_data)

                total_iemg += iemg
                total_mav += iemg / window_size
                total_ssi += np.sum(sq_window_data)
            
            avg_iemg[channel, epoch] = total_iemg / num_windows
            avg_mav[channel, epoch] = total_mav / num_windows
            avg_ssi[channel, epoch] = total_ssi / num_windows

    return avg_iemg, avg_mav, avg_ssi


### Global Variables
folder_name = "Data/"
in_file_path = folder_name + "hand_fist.txt"
out_file_path = folder_name + "hand_fist.csv"
fs = 250


### Main function
def main():

    ## For each trial, right now hand gist only, loop through each trial once have start time for all trials

    # Create csv file
    txt_2_csv(in_file_path, out_file_path)

    data_raw = pd.read_csv(out_file_path)

    t = np.arange(1/fs, len(data_raw)/fs + 1/fs,1/fs)

    # col_headers = ['EXGChannel0', 'EXGChannel1', 'EXGChannel2', 'EXGChannel3', 'EXGChannel4', 'EXGChannel5', 'EXGChannel6', 'EXGChannel7']
    col_headers = ['EXGChannel0', 'EXGChannel1', 'EXGChannel2', 'EXGChannel3']

    low_cut = 5
    high_cut = 90

    # Apply band pass filter to channel 0 - 4
    data_filt = filt_data(data_raw, col_headers, fs, low_cut, high_cut)

    # 3D matrix: 1st axis = channels #; 2nd axis = sample #; 3rd axis = epoch #
    all_channels = []
    for channel in col_headers:
        all_channels.append(epoch_data(10.08, data_filt[channel], t, 1.5))

    # extract outliers that are 8 times the average value from the 3D matrix: 
    data_trimmed = extract_outlier_epochs(all_channels, 8)

    # Feature extractions for IEMG, MAV, and SSI
    # set rolling window size to be 5 samples
    window_size = 5
    #  each output is a 2D matrix where: first axis = channel #; second axis = epoch # 
    iemg, mav, ssi = calc_time_features(data_trimmed, window_size)


    ## Plot graphs
    plot_time_series(data_filt)



if __name__ == "__main__":
    main()