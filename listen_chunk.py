from pylsl import StreamInlet, resolve_stream
import numpy as np
import threading
from preprocess_data_with_epoch_extraction import preprocess_data 
from extract_features import real_time_calc_features
from find_MVC import calculate_MVC_realtime
import pickle

mvc_list = np.array([-723.0118778958175, -97.4759575695941, -2055.220550921029, 2011.522890510904, 4650.414544393367, 
 4772.410365630432, 1688.7189970873592, -2478.160259504349])

svm_model = pickle.load(open('svm_model.pkl', 'rb'))


### Settings:
# On OpenBCI GUI, set widget to Networking
# set Protocol to LSL 
# Stream 1: Data Type = TimeSeriesRaw; Type 'EMG' 
# Stream 2: Data Type = Accel/Aux; Type 'Accel/Aux' 

### OpenBCI: Start Data Stream, Start LSL Stream

### Run this script on terminal, and wait...

### Outcome:
# Resulted samples displayed in terminal are in the following 2 formats:
# https://docs.google.com/document/d/e/2PACX-1vR_4DXPTh1nuiOwWKwIZN3NkGP3kRwpP4Hu6fQmy3jRAOaydOuEI1jket6V4V6PG4yIG15H1N7oFfdV/pub

# 1. EMG stream: 
# A msg is sent each frame, each msg consists of samples from each channel:
# [channel_1_sample_1, channel_2_sample_1, channel_3_sample_1, channel_4_sample_1], timestamp_1
# [channel_1_sample_2, channel_2_sample_2, channel_3_sample_2, channel_4_sample_2], timestamp_2
# [channel_1_sample_3, channel_2_sample_3, channel_3_sample_3, channel_4_sample_3], timestamp_3
# ...

# 2. Accel/Aux stream:
# Three floats one for each axis: 
# [x, y, z], timestamp

### Termination:
# This script stops listening for data from the network when:
# 1. Keyboard interruption: Ctrl C in terminal
# 2. Closing terminal window
# 3. Source of stream stops
# 4. Unhandled exception occurs



### Main function:
def main():
    try:
        # First resolve an EMG stream on the lab network
        print("Looking for an EMG stream...")
        streams = resolve_stream('type', 'EMG')

        # Create a new inlet to read from the stream
        inlet = StreamInlet(streams[0], max_chunklen=12)
        eeg_time_correction = inlet.time_correction()

        info = inlet.info()  # Define the 'info' object
        description = info.desc()
        fs = int(info.nominal_srate())  # Sampling rate
        print("Sampling rate: ", fs)

        second_buffer = np.zeros((fs*1.5,8))  # Buffer for one second of data
        print(second_buffer)

        while True:
            # Get a new sample
            chunk, timestamps = inlet.pull_chunk(timeout=1.5, max_samples=fs)

            if timestamps:
                print("New chunk!")
                len_chunk = len(chunk)
                print("Length: ", len(chunk), "Width: ", len(chunk[0]), "Type: ", type(chunk))
                
                if len_chunk < fs: # May lost some sample in the first chunk
                    print("Incomplete chunk window")

                else:
                    second_buffer[:] = np.array(chunk)
                    # second_buffer = np.concatenate((second_buffer, np.array(chunk).flatten()))[-fs:]
                    preproc_data = preprocess_data(second_buffer,fs)/mvc_list
                    combined_features = real_time_calc_features(preproc_data)

                    predictions = svm_model.predict(combined_features)

                 
                    print(second_buffer)

                    # if len(second_buffer) >= fs:
                    #     # Now 'second_buffer' contains one second of data
                    #     # Process this data
                    #     pass
    
    except KeyboardInterrupt:
        print("\nStream reading interrupted by user (Ctrl+C). Exiting...")


if __name__ == "__main__":
    main()
