#input trimmed_data

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

    # add to the end of last for loop in preprocess and declare variances as new variable in for loop:
    #trial_variance = get_var_trial(trimmed_data)
    #variances.append(trial_variance)
        