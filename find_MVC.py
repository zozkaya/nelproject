## file for classifier code 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

path = '/Users/zeynepozkaya/Desktop/nov21/MVC'

def calculate_MVC(path):
    trial_dict = {}
    #input: file path of the MVCs
    #return: dictionary of max contract value for each channel and each task
    for trial in os.listdir(path):
        #extract only file ending in .txt (for if on mac)
        if '.txt' in trial:
            #reach in file
            data=pd.read_csv(os.path.join(path,trial), header=0,sep=',',skiprows=4)
            #find maximum
            trial_dict[trial.replace('.txt','')] = data[[col for col in data if 'EXG' in col]].iloc[50:].max()

    return trial_dict

def calculate_MVC_realtime(MVC_dict):
    trial_dict = {}
    #input: file path of the MVCs
    #return: dictionary of max contract value for each channel and each task

    chan_0 = max(MVC_dict['index_finger_point_MVC'][0],MVC_dict['hand_open_MVC'][0],
                 MVC_dict['wrist_down_MVC'][0],MVC_dict['hand_fist_MVC'][0],
                 MVC_dict['wrist_left_MVC'][0],MVC_dict['wrist_up_MVC'][0],
                 MVC_dict['wrist_right_MVC'][0],MVC_dict['two_finger_pinch_MVC'][0])

    chan_1 = max(MVC_dict['index_finger_point_MVC'][1],MVC_dict['hand_open_MVC'][1],
                 MVC_dict['wrist_down_MVC'][1],MVC_dict['hand_fist_MVC'][1],
                 MVC_dict['wrist_left_MVC'][1],MVC_dict['wrist_up_MVC'][1],
                 MVC_dict['wrist_right_MVC'][1],MVC_dict['two_finger_pinch_MVC'][1])
    
    chan_2 = max(MVC_dict['index_finger_point_MVC'][2],MVC_dict['hand_open_MVC'][2],
                 MVC_dict['wrist_down_MVC'][2],MVC_dict['hand_fist_MVC'][2],
                 MVC_dict['wrist_left_MVC'][2],MVC_dict['wrist_up_MVC'][2],
                 MVC_dict['wrist_right_MVC'][2],MVC_dict['two_finger_pinch_MVC'][2])
    
    chan_3 = max(MVC_dict['index_finger_point_MVC'][3],MVC_dict['hand_open_MVC'][3],
                 MVC_dict['wrist_down_MVC'][3],MVC_dict['hand_fist_MVC'][3],
                 MVC_dict['wrist_left_MVC'][3],MVC_dict['wrist_up_MVC'][3],
                 MVC_dict['wrist_right_MVC'][3],MVC_dict['two_finger_pinch_MVC'][3])
    
    chan_4 = max(MVC_dict['index_finger_point_MVC'][4],MVC_dict['hand_open_MVC'][4],
                 MVC_dict['wrist_down_MVC'][4],MVC_dict['hand_fist_MVC'][4],
                 MVC_dict['wrist_left_MVC'][4],MVC_dict['wrist_up_MVC'][4],
                 MVC_dict['wrist_right_MVC'][4],MVC_dict['two_finger_pinch_MVC'][4])
    
    chan_5 = max(MVC_dict['index_finger_point_MVC'][5],MVC_dict['hand_open_MVC'][5],
                 MVC_dict['wrist_down_MVC'][5],MVC_dict['hand_fist_MVC'][5],
                 MVC_dict['wrist_left_MVC'][5],MVC_dict['wrist_up_MVC'][5],
                 MVC_dict['wrist_right_MVC'][5],MVC_dict['two_finger_pinch_MVC'][5])
    
    chan_6 = max(MVC_dict['index_finger_point_MVC'][6],MVC_dict['hand_open_MVC'][6],
                 MVC_dict['wrist_down_MVC'][6],MVC_dict['hand_fist_MVC'][6],
                 MVC_dict['wrist_left_MVC'][6],MVC_dict['wrist_up_MVC'][6],
                 MVC_dict['wrist_right_MVC'][6],MVC_dict['two_finger_pinch_MVC'][6])
    
    chan_7 = max(MVC_dict['index_finger_point_MVC'][7],MVC_dict['hand_open_MVC'][7],
                 MVC_dict['wrist_down_MVC'][7],MVC_dict['hand_fist_MVC'][7],
                 MVC_dict['wrist_left_MVC'][7],MVC_dict['wrist_up_MVC'][7],
                 MVC_dict['wrist_right_MVC'][7],MVC_dict['two_finger_pinch_MVC'][7])
  
    mvc_list = [chan_0 ,chan_1 ,chan_2, chan_3, chan_4, chan_5 ,chan_6 ,chan_7]

    return mvc_list

