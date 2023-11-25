#input trimmed_data

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

# add to the end of last for loop in preprocess and declare rms as new variable before for loop:
# trial_rms = get_rms_trial(trimmed_data)
# rms.append(trial_rms)