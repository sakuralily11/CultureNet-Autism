from deepNet import deepNet
from utils import load_data, leave_1_out_ids, all_children_ids, target_only_ids
from models import *

from keras.models import Model
from keras.layers import Dense, Input, concatenate, BatchNormalization, Activation, Dropout
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, CSVLogger
from keras import backend as K
import keras
import numpy as np
import pickle as pkl

# Control randomness
import os
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
np.random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import random as rn
rn.seed(12345)
tf.set_random_seed(1234)

if __name__ == '__main__':

    c0_IDs = [1,2,3,4,6,7,8,9,10,11,12,13,14,16,17] # Culture index 0
    c1_IDs = [2,3,4,5,6,7,8,9,10,13,14,15,17,18,20] # Culture index 1

    c0_IDs_1Out = leave_1_out_ids(c0_IDs)
    c1_IDs_1Out = leave_1_out_ids(c1_IDs)

    c0_IDs_All = all_children_ids(c0_IDs)
    c1_IDs_All = all_children_ids(c1_IDs)

    c0_IDs_Target = target_only_ids(c0_IDs)
    c1_IDs_Target = target_only_ids(c1_IDs)

    # Loop over folds 
    for i in range(len(c0_IDs_1Out)): 

        c0_data = load_data(c0_IDs_1Out[i], 0, data_proportion=[1,0,0.2,0.8])
        c1_data = load_data(c1_IDs_1Out[i], 1, data_proportion=[1,0,0.2,0.8])

        c0_data_All20 = load_data(c0_IDs_All[i], 0, data_proportion=[0.2,0,0.2,0.8])
        c1_data_All20 = load_data(c1_IDs_All[i], 1, data_proportion=[0.2,0,0.2,0.8])

        c0_data_Target20 = load_data(c0_IDs_Target[i], 0, data_proportion=[0.2,0,0.2,0.8])
        c1_data_Target20 = load_data(c1_IDs_Target[i], 1, data_proportion=[0.2,0,0.2,0.8])

        # Note: Increase iterations as needed (for validation)
        for loop in range(1): 

            print('---------- FOLD %s ----------'%(i+1))
            print('---------- LOOP %s ----------'%(loop+1))

            """ 
            Model 1 - Within Culture / SI: 
            Train and test on each culture 
            """ 
            run_m1(c0_data, c1_data, 'c0_m1', 'c1_m1')

            """
            Model 2 - Between Culture / SI: 
            Train on culture A, test on culture B 
            """
            run_m2(c0_data, c1_data, 'c0_m2', 'c1_m2')

            """
            Model 3 - Mixed Culture / SI: 
            Train on both cultures, test on each culture 
            """
            run_m3(c0_IDs, c1_IDs, c0_IDs_1Out, c1_IDs_1Out, i)

            """
            Model 4 - Joint Culture / SI (CultureNet): 
            Train on both cultures, fine tune with culture A, test on culture A 
            """
            run_m4(c0_data, c1_data)

            """
            Model 5 - Joint Culture / SD (GenNet): 
            Train and test on each culture, including 20% of target data 
            """
            run_m5(c0_data_All20, c1_data_All20)

            """
            Model 6 - Individual / SD: 
            Train and test on each culture, using only 20% of target data 
            """
            run_m6(c0_data_Target20, c1_data_Target20)

            """
            Model 7 - Joint Culture / SD: 
            Train on both cultures, fine tune with culture A, fine tune with target data, test on culture A 
            """
            run_m7(c0_IDs, c1_IDs, c0_IDs_Target, c1_IDs_Target)