## file for classifier code 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

<<<<<<< HEAD
=======
path = '/Users/zeynepozkaya/Desktop/MVC'

>>>>>>> cc9e6b4a994a595bcd2594fed6ea3a1dfd7542aa
def calculate_MVC(path):
    trial_dict = {}
    #input: file path of the MVCs
    #return: dictionary of max contract value for each channel and each task
    for trial in os.listdir(path):
        #extract only file ending in .txt (for if on mac)
        if '.txt' in trial:
            #reach in file
            data=pd.read_csv(os.path.join(path,trial), header=0,sep=',',skiprows=4)
<<<<<<< HEAD
            #find maximum
            trial_dict[trial.replace('.txt','')] = data[[col for col in data if 'EXG' in col]].max()
            
    return trial_dict
=======

            #find maximum
            trial_dict[trial.replace('.txt','')] = data[[col for col in data if 'EXG' in col]].iloc[50:].max()
    
    return trial_dict

>>>>>>> cc9e6b4a994a595bcd2594fed6ea3a1dfd7542aa
