from pylsl import StreamInlet, resolve_stream
import threading

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



### Functions
# read_stream(): handles the reading of real-time data of EMG and Accel/Aux
# Input:
# stream: the stream to be read, obtained by using resolve_stream from pylsl
# stream_name: a string of the name of the type of stream, for identification
def read_stream(stream, stream_name):
    inlet = StreamInlet(stream)
    print(f"Started reading from stream {stream_name}")

    try:
        while True:
            sample, timestamp = inlet.pull_sample()
            print(f"Stream {stream_name}: {sample}, Timestamp: {timestamp}")

    except KeyboardInterrupt:
        print("\nStream reading interrupted by user (Ctrl+C). Exiting...")



### Main function:
def main():
    try:
        print("Looking for an EEG stream...")
        
        # finds streams of type "EMG" on the network and stores them in emg_streams
        emg_streams = resolve_stream('type', 'EMG')
        accel_aux_streams = resolve_stream('type', 'Accel/Aux')

        # Checks if both EMG and Accel/Aux streams are available
        if emg_streams and accel_aux_streams:
            # Create Thread objects for EMG and AUX streams
            emg_thread = threading.Thread(target=read_stream, args=(emg_streams[0], 'EMG'))
            accel_aux_thread = threading.Thread(target=read_stream, args=(accel_aux_streams[0], 'Accel/AUX'))

            # Start the threads
            emg_thread.start()
            accel_aux_thread.start()

            # Wait for the threads to finish (they won't, unless you send a KeyboardInterrupt)
            emg_thread.join()
            accel_aux_thread.join()

        else:
            print("EMG and/or AUX enough EEG streams found.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()