from deepNet import deepNet
from utils import load_data

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

def __run_deep_net(data, dtype, trainable=[True,True,True,True,True], weights=None, **kwargs):
    """ Builds, trains, and tests basic deep model 
        Returns model weights 

    PARAMETERS 
    data: tuple of loaded data 
    dtype: string ID 
    """

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    _, _, id_test, x_train, x_val, x_test, y_train, y_val, y_test, _, _, culture_test, _, _, frame_test = data

    # Build model 
    model = deepNet(input_dim=x_train.shape[1], trainable=trainable)
    if weights is None: 
        model.train_model(x_train, y_train, x_val, y_val, dtype)
    else: 
        model.set_weights(weights)
    _ = model.make_report('ExperimentdS_deep_'+dtype, id_test, x_test, y_test, culture_test, frame_test)

    optimized_weights = model.get_weights()
    
    K.clear_session()

    return optimized_weights  

def __run_deep_joint(data_cA, data_cB, prelim_data, dtype_prelim_cA, dtype_final_cA):
    """ Builds, trains, and tests joint deep model 

    PARAMETERS 
    data_cA: tuple of loaded data for culture A (culture being tested)
    data_cB: tuple of loaded data for culture B 
    prelim_data: tuple of merged data for preliminary model 
    dtype_prelim_cA: string ID for culture A, preliminary results 
    dtype_final_cA: string ID for culture A, preliminary results 
    """ 

    # Build preliminary model 
    prelim_weights = _run_deep_net(prelim_data, dtype_prelim_cA)

    # Build culture-specific model (for culture A)
    _ = _run_deep_net(data_cA, dtype_final_cA, trainable=[False,False,False,False,True], weights=prelim_weights)

    return None 

def run_m1(c0_data, c1_data, c0_dtype, c1_dtype):
    """ 
    Runs Model 1 - Within Culture / SI 
    Train and test on each culture 

    PARAMETERS 
    c0_data: tuple of loaded data for culture 0 
    c1_data: tuple of loaded data for culture 1 
    c0_dtype: string ID for culture 0 
    c1_dtype: string ID for culture 1 
    """ 

    print('---------- Running Model 1 ----------')

    _ = __run_deep_net(c0_data, c0_dtype)
    _ = __run_deep_net(c1_data, c1_dtype)

    print('---------- Completed Model 1 ----------')

    return None 

def run_m2(c0_data, c1_data, c0_dtype, c1_dtype): 
    """ 
    Runs Model 2 - Between Culture / SI 
    Train on culture A, test on culture B 

    PARAMETERS 
    c0_data: tuple of loaded data for culture 0 
    c1_data: tuple of loaded data for culture 1 
    c0_dtype: string ID for culture 0 
    c1_dtype: string ID for culture 1 
    """ 

    print('---------- Running Model 2 ----------')

    """ Train on Serbia, Validate on Japan, Test on Japan """
    c0_m2_data = (c1_data[0], c0_data[1], c0_data[2], c1_data[3], c0_data[4], c0_data[5], c1_data[6], c0_data[7], c0_data[8], c1_data[9], c0_data[10], c0_data[11], c1_data[12], c0_data[13], c0_data[14])
    _ = __run_deep_net(c0_m2_data, c0_dtype)

    """ Train on Japan, Validate on Serbia, Test on Serbia """
    c1_m2_data = (c0_data[0], c1_data[1], c1_data[2], c0_data[3], c1_data[4], c1_data[5], c0_data[6], c1_data[7], c1_data[8], c0_data[9], c1_data[10], c1_data[11], c0_data[12], c1_data[13], c1_data[14])
    _ = __run_deep_net(c1_m2_data, c1_dtype)

    print('---------- Completed Model 2 ----------')

    return None

def run_m3(c0_IDs, c1_IDs, c0_IDs_1Out, c1_IDs_1Out, fold):
    """ 
    Runs Model 3 - Mixed Culture / SI 
    Train on both cultures, test on each culture 

    PARAMETERS 
    c0_IDs: list of IDs for culture 0 
    c1_IDs: list of IDs for culture 1 
    c0_IDs_1Out: list of leave-1-out IDs for culture 0 
    c1_IDs_1Out: list of leave-1-out IDs for culture 1 
    fold: int of current fold 
    """ 

    print('---------- Running Model 3 ----------')

    culture = [0,1]

    # Remove test IDs from list 
    c0_m3_IDs = list(set(c0_IDs).difference(set(c0_IDs_1Out[fold][2])))
    c1_m3_IDs = list(set(c1_IDs).difference(set(c1_IDs_1Out[fold][2])))

    c0_m3_train_IDs = []
    c1_m3_train_IDs = []

    # Loop 10 times to choose train data 
    for _ in range(10): 
        c = random.choice(culture) 
        if c == 0: 
            rand_train = random.choice(c0_m3_IDs)
            c0_m3_train_IDs.append(rand_train)
            c0_m3_IDs.remove(rand_train)
        else: 
            rand_train = random.choice(c1_m3_IDs)
            c1_m3_train_IDs.append(rand_train)
            c1_m3_IDs.remove(rand_train)

    c0_test_IDs = c0_IDs_1Out[fold][2]
    c1_test_IDs = c1_IDs_1Out[fold][2]
    c0_val_IDs = c0_IDs_1Out[fold][1]
    c1_val_IDs = c1_IDs_1Out[fold][1]

    c0_m3_data_IDs = [c0_m3_train_IDs, c0_val_IDs, c0_test_IDs]
    c1_m3_data_IDs = [c1_m3_train_IDs, c1_val_IDs, c1_test_IDs]

    c0_m3_data = load_data(c0_m3_data_IDs, 0, data_proportion=[1,0,0.2,0.8])
    c1_m3_data = load_data(c1_m3_data_IDs, 1, data_proportion=[1,0,0.2,0.8])

    # Merge datasets  
    merged_data = []
    for q in range(len(c0_m3_data)):
        if c0_m3_data[q] is None:
            merged_data.append(c1_m3_data[q])
        elif c1_m3_data[q] is None:
            merged_data.append(c0_m3_data[q])
        else: 
            merged_data.append(np.concatenate((c0_m3_data[q], c1_m3_data[q]), axis=0))
    merged_data = tuple(merged_data)

    _ = __run_deep_net(merged_data, 'm3') 

    print('---------- Completed Model 3 ----------')

    return None

def run_m4(c0_data, c1_data): 
    """ 
    Runs Model 4 - Joint Culture / SI (CultureNet)
    Train on both cultures, fine tune with culture A, test on culture A 

    PARAMETERS 
    c0_data: tuple of loaded data for culture 0 
    c1_data: tuple of loaded data for culture 1 
    """ 

    print('---------- Running Model 4 ----------')

    """ Train on both cultures, fine tune and test on Japan """

    # Merge culture data for initial training (test & val on Japan)
    id_train = np.concatenate((c0_data[0], c1_data[0], c1_data[1], c1_data[2]), axis=0)
    id_val, id_test = c0_data[1], c0_data[2]

    x_train = np.concatenate((c0_data[3], c1_data[3], c1_data[4], c1_data[5]), axis=0)
    x_val, x_test = c0_data[4], c0_data[5]

    y_train = np.concatenate((c0_data[6], c1_data[6], c1_data[7], c1_data[8]), axis=0)
    y_val, y_test = c0_data[7], c0_data[8]

    culture_train = np.concatenate((c0_data[9], c1_data[9], c1_data[10], c1_data[11]), axis=0)
    culture_val, culture_test = c0_data[10], c0_data[11]

    frame_train = np.concatenate((c0_data[12], c1_data[12], c1_data[13], c1_data[14]), axis=0)
    frame_val, frame_test = c0_data[13], c0_data[14]

    prelim_data = (id_train, id_val, id_test, x_train, x_val, x_test, y_train, y_val, y_test, culture_train, culture_val, culture_test, frame_train, frame_val, frame_test)

    __run_deep_joint(c0_data, c1_data, prelim_data, 'c0_m4_prelim', 'c0_m4_final')

    """ Train on both cultures, fine tune and test on Serbia """ 

    # Merge culture data for initial training (test & val on Serbia)
    id_train = np.concatenate((c1_data[0], c0_data[0], c0_data[1], c0_data[2]), axis=0)
    id_val, id_test = c1_data[1], c1_data[2]

    x_train = np.concatenate((c1_data[3], c0_data[3], c0_data[4], c0_data[5]), axis=0)
    x_val, x_test = c1_data[4], c1_data[5]

    y_train = np.concatenate((c1_data[6], c0_data[6], c0_data[7], c0_data[8]), axis=0)
    y_val, y_test = c1_data[7], c1_data[8]

    culture_train = np.concatenate((c1_data[9], c0_data[9], c0_data[10], c0_data[11]), axis=0)
    culture_val, culture_test = c1_data[10], c1_data[11]

    frame_train = np.concatenate((c1_data[12], c0_data[12], c0_data[13], c0_data[14]), axis=0)
    frame_val, frame_test = c1_data[13], c1_data[14]

    prelim_data = (id_train, id_val, id_test, x_train, x_val, x_test, y_train, y_val, y_test, culture_train, culture_val, culture_test, frame_train, frame_val, frame_test)

    __run_deep_joint(c1_data, c0_data, prelim_data, 'c1_m4_prelim', 'c1_m4_final')

    print('---------- Completed Model 4 ----------')

    return None

def run_m5(c0_data_All20, c1_data_All20): 
    """ 
    Runs Model 5 - Joint Culture / SD (GenNet) 
    Train and test on each culture, including 20% of target data 

    PARAMETERS 
    c0_data_All20: tuple of loaded data for culture 0 (includes 20% of target data)
    c1_data_All20: tuple of loaded data for culture 1 (includes 20% of target data)
    """ 

    print('---------- Running Model 5 ----------')

    """ Train and test on Japan """ 
    _ = __run_deep_net(c0_data_All20, 'c0_m5')

    """ Train and test on Japan """ 
    _ = __run_deep_net(c1_data_All20, 'c1_m5')

    print('---------- Completed Model 5 ----------')

    return None

def run_m6(c0_data_Target20, c1_data_Target20): 
    """ 
    Runs Model 6 - Individual / SD 
    Train and test on each culture, using only 20% of target data 

    PARAMETERS 
    c0_data_Target20: tuple of loaded data for culture 0 (only 20% of target data)
    c1_data_Target20: tuple of loaded data for culture 1 (only 20% of target data)
    """ 

    print('---------- Running Model 6 ----------')

    """ Train and test on Japan """ 
    _ = __run_deep_net(c0_data_Target20, 'c0_m6')

    """ Train and test on Japan """ 
    _ = __run_deep_net(c1_data_Target20, 'c1_m6')

    print('---------- Completed Model 6 ----------')

    return None

def run_m7(c0_IDs, c1_IDs, c0_IDs_Target, c1_IDs_Target): 
    """ 
    Runs Model 7 - Joint Culture / SD 
    Train on both cultures, fine tune with culture A, fine tune with target data, test on culture A 

    PARAMETERS 
    c0_IDs: list of IDs for culture 0 
    c1_IDs: list of IDs for culture 1 
    c0_IDs_Target: list of target child IDs for culture 0 
    c1_IDs_Target: list of target child IDs for culture 1 
    """ 

    print('---------- Running Model 7 ----------')

    c0_m7_data_IDs = [c0_IDs]*3
    c1_m7_data_IDs = [c1_IDs]*3

    c0_m7_data = load_data(c0_m7_data_IDs, 0, data_proportion=[0.2,0,0.2,0.8])
    c1_m7_data = load_data(c1_m7_data_IDs, 1, data_proportion=[0.2,0,0.2,0.8])

    # Part A: Joint Model (train & validate on 20% of all children) 
    joint_data = []
    for p in range(len(c0_m7_data)):
        joint_data.append(np.concatenate((c0_m7_data[p], c1_m7_data[p]), axis=0))
    joint_data = tuple(joint_data)

    prelim_weights = _run_deep_net(joint_data, 'm7_prelim')

    # Part B: Culture Specific Model (train & validate on 20% of all children)
    c0_m7_prelim_weights = __run_deep_net(c0_m7_data, 'c0_m7_prelim', trainable=[False,False,False,False,True], weights=prelim_weights)
    c1_m7_prelim_weights = __run_deep_net(c1_m7_data, 'c1_m7_prelim', trainable=[False,False,False,False,True], weights=prelim_weights)

    # Loop over 15 folds 
    for q in range(len(c0_IDs_Target)):
        c0_child_data = load_data(c0_IDs_Target[q], 0, data_proportion=[0.2,0,0.2,0.8])
        c1_child_data = load_data(c1_IDs_Target[q], 1, data_proportion=[0.2,0,0.2,0.8])

        for r in range(10):
            _ = __run_deep_net(c0_child_data, 'c0_m7_ID%s'%(c0_IDs[q]), trainable=[False,False,False,False,True], weights=c0_m7_prelim_weights)
            _ = __run_deep_net(c1_child_data, 'c1_m7_ID%s'%(c1_IDs[q]), trainable=[False,False,False,False,True], weights=c1_m7_prelim_weights)

    print('---------- Completed Model 7 ----------')

    return None
