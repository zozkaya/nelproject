from pylsl import StreamInlet, resolve_stream
import numpy as np
from preprocess_data_with_epoch_extraction import preprocess_data
from extract_features import calc_features

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
        inlet = StreamInlet(streams[0])
        eeg_time_correction = inlet.time_correction()

        info = inlet.info()  # Define the 'info' object
        description = info.desc()
        fs = int(info.nominal_srate())  # Sampling rate
        print("Sampling rate: ", fs)

        buffer_length = int(fs * 1.5)
        buffer_window = np.zeros((buffer_length,8))  # Buffer for one second of data
        print(buffer_window)

        while True:
            # Get a new sample
            chunk, timestamps = inlet.pull_chunk(timeout=1.5, max_samples=buffer_length)

            if timestamps:
                print("New chunk!")
                len_chunk = len(chunk)
                print("Length: ", len(chunk), "Width: ", len(chunk[0]), "Type: ", type(chunk))
                
                chunk_array = np.array(chunk)
                if len_chunk > buffer_length:
                    chunk_array = chunk_array[:buffer_length]  # Trim extra samples
                elif len_chunk < buffer_length:
                    # Pad with zeros if chunk is smaller than expected
                    chunk_array = np.pad(chunk_array, ((0, buffer_length - len_chunk), (0, 0)), 'constant')

                buffer_window[:] = chunk_array
                prepross_data = preprocess_data(buffer_window, fs)
                
                # extract_features = calc_features(buffer_window, 0, 250)
                
                print(buffer_window)
                print(prepross_data)


                    # if len(second_buffer) >= fs:
                    #     # Now 'second_buffer' contains one second of data
                    #     # Process this data
                    #     pass
    
    except KeyboardInterrupt:
        print("\nStream reading interrupted by user (Ctrl+C). Exiting...")


if __name__ == "__main__":
    main()
