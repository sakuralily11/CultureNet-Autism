from deepNet import deepNet
from utils import load_data, leave_1_out_ids, all_children_ids, target_only_ids, merge_data
from models import *

from keras.models import Model
from keras.layers import Dense, Input, concatenate, BatchNormalization, Activation, Dropout
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, CSVLogger
from keras import backend as K
import keras
import os
import pathlib
import tensorflow as tf
import numpy as np
import pickle as pkl

# Control randomness
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
np.random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import random as rn
rn.seed(12345)
tf.set_random_seed(1234)

if __name__ == '__main__':

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPORTS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'Reports')
    WEIGHTS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'Weights')
    pathlib.Path(REPORTS_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(WEIGHTS_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 

    c0_IDs = [1,2,3,4,6,7,8,9,10,11,12,13,14,16,17] # Culture index 0
    c1_IDs = [2,3,4,5,6,7,8,9,10,13,14,15,17,18,20] # Culture index 1

    c0_IDs_1Out = leave_1_out_ids(c0_IDs)
    c1_IDs_1Out = leave_1_out_ids(c1_IDs)

    c0_IDs_All = [c0_IDs]*3
    c1_IDs_All = [c1_IDs]*3

    c0_IDs_targetRep = all_children_ids(c0_IDs)
    c1_IDs_targetRep = all_children_ids(c1_IDs)

    c0_IDs_targetOnly = target_only_ids(c0_IDs)
    c1_IDs_targetOnly = target_only_ids(c1_IDs)

    c0_data_All = load_data(c0_IDs_All, 0, data_proportion=[0.2,0,0.2,0.8])
    c1_data_All = load_data(c1_IDs_All, 1, data_proportion=[0.2,0,0.2,0.8])
    
    print('done')

    m7_joint_data = []
    for p in range(len(c0_data_All)):
        m7_joint_data.append(np.concatenate((c0_data_All[p], c1_data_All[p]), axis=0))
    m7_joint_data = tuple(m7_joint_data)

    """
    Preliminary Model 7 - Joint Culture / SD: 
    Train on both cultures, fine tune with culture A 
    """
    c0_m7_prelim_weights, c1_m7_prelim_weights = run_prelim_m7(m7_joint_data, c0_data_All, c1_data_All)

    # Loop over target children  
    for i in range(len(c0_IDs_1Out)): 

        c0_data = load_data(c0_IDs_1Out[i], 0, data_proportion=[0.8,0.8,1,0.8])
        c1_data = load_data(c1_IDs_1Out[i], 1, data_proportion=[0.8,0.8,1,0.8])

        c0_data_targetRep = load_data(c0_IDs_targetRep[i], 0, data_proportion=[0.2,0,0.2,0.8])
        c1_data_targetRep = load_data(c1_IDs_targetRep[i], 1, data_proportion=[0.2,0,0.2,0.8])

        c0_data_targetOnly = load_data(c0_IDs_targetOnly[i], 0, data_proportion=[0.2,0,0.2,0.8])
        c1_data_targetOnly = load_data(c1_IDs_targetOnly[i], 1, data_proportion=[0.2,0,0.2,0.8])

        c0_data_merged = merge_data(c0_data, c1_data)
        c1_data_merged = merge_data(c1_data, c0_data)

        # 10-fold k-validation 
        for loop in range(10): 

            print('---------- CHILD {} ----------'.format(i+1))
            print('---------- FOLD {} ----------'.format(loop+1))

            c0_m3_weights = None 
            c1_m3_weights = None 

            """ 
            Model 1 - Within Culture / SI: 
            Train and test on each culture 
            """ 
            run_m1(c0_data, c1_data)

            """
            Model 2 - Between Culture / SI: 
            Train on culture A, test on culture B 
            """
            run_m2(c0_data, c1_data)

            """
            Model 3 - Mixed Culture / SI: 
            Train on both cultures, test on each culture 
            """
            c0_m3_weights, c1_m3_weights = run_m3(c0_data_merged, c1_data_merged)

            """
            Model 4 - Joint Culture / SI (CultureNet): 
            Train on both cultures, fine tune with culture A, test on culture A 
            """
            run_m4(c0_data, c1_data, c0_data_merged, c1_data_merged, c0_m3_weights, c1_m3_weights)

            """
            Model 5 - Joint Culture / SD (GenNet): 
            Train and test on each culture, including 20% of target data 
            """
            run_m5(c0_data_targetRep, c1_data_targetRep)

            """
            Model 6 - Individual / SD: 
            Train and test on each culture, using only 20% of target data 
            """
            run_m6(c0_data_targetOnly, c1_data_targetOnly)

            """
            Model 7 - Joint Culture / SD: 
            Train on both cultures (prelim), fine tune with culture A (prelim), fine tune with target data, test on culture A 
            """
            run_m7(c0_data_targetOnly, c1_data_targetOnly, c0_m7_prelim_weights, c1_m7_prelim_weights)

    # Process Data 