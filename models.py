from deepNet import deepNet

from keras import backend as K
import os
import numpy as np
import tensorflow as tf

# Control randomness
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
np.random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import random as rn 
rn.seed(12345)
tf.set_random_seed(1234)

def __run_deep_net(data, dtype, trainable=[True,True,True,True,True], weights=None, **kwargs):
    """ 
    Builds, trains, and tests basic deep model 
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
        model.train_model(x_train, y_train, x_val, y_val, dtype)
    _ = model.make_report('ExperimentdS_deep_'+dtype, id_test, x_test, y_test, culture_test, frame_test)

    optimized_weights = model.get_weights()
    
    K.clear_session()

    return optimized_weights  

def __run_deep_joint(data_cA, data_cB, prelim_data, dtype_prelim_cA, dtype_final_cA):
    """ 
    Builds, trains, and tests joint deep model 

    PARAMETERS 
    data_cA: tuple of loaded data for culture A (culture being tested)
    data_cB: tuple of loaded data for culture B 
    prelim_data: tuple of merged data for preliminary model 
    dtype_prelim_cA: string ID for culture A, preliminary results 
    dtype_final_cA: string ID for culture A, preliminary results 
    """ 

    # Build preliminary model 
    prelim_weights = __run_deep_net(prelim_data, dtype_prelim_cA)

    # Build culture-specific model (for culture A)
    _ = __run_deep_net(data_cA, dtype_final_cA, trainable=[False,False,False,False,True], weights=prelim_weights)

    return None 

def run_m1(c0_data, c1_data):
    """ 
    Runs Model 1 - Within Culture / SI 
    Train and test on each culture 

    PARAMETERS 
    c0_data: tuple of loaded data for culture 0 
    c1_data: tuple of loaded data for culture 1 
    """ 

    print('---------- Running Model 1 ----------')

    _ = __run_deep_net(c0_data, 'm1')
    _ = __run_deep_net(c1_data, 'm1')

    print('---------- Completed Model 1 ----------')

    return None 

def run_m2(c0_data, c1_data): 
    """ 
    Runs Model 2 - Between Culture / SI 
    Train on culture A, test on culture B 

    PARAMETERS 
    c0_data: tuple of loaded data for culture 0 
    c1_data: tuple of loaded data for culture 1 
    """ 

    print('---------- Running Model 2 ----------')

    """ Train on Serbia, Validate on Japan, Test on Japan """
    c0_m2_data = (c1_data[0], c0_data[1], c0_data[2], c1_data[3], c0_data[4], c0_data[5], c1_data[6], c0_data[7], c0_data[8], c1_data[9], c0_data[10], c0_data[11], c1_data[12], c0_data[13], c0_data[14])
    _ = __run_deep_net(c0_m2_data, 'm2')

    """ Train on Japan, Validate on Serbia, Test on Serbia """
    c1_m2_data = (c0_data[0], c1_data[1], c1_data[2], c0_data[3], c1_data[4], c1_data[5], c0_data[6], c1_data[7], c1_data[8], c0_data[9], c1_data[10], c1_data[11], c0_data[12], c1_data[13], c1_data[14])
    _ = __run_deep_net(c1_m2_data, 'm2')

    print('---------- Completed Model 2 ----------')

    return None

def run_m3(c0_data_merged, c1_data_merged):
    """ 
    Runs Model 3 - Mixed Culture / SI 
    Train on both cultures, test on each culture 
    Returns model weights 

    PARAMETERS 
    c0_data_merged: tuple of merged data, with culture 0 as target 
    c1_data_merged: tuple of merged data, with culture 1 as target 
    """ 

    print('---------- Running Model 3 ----------')

    """ Train on both cultures, test on Japan """

    c0_m3_weights = __run_deep_net(c0_data_merged, 'm3') 

    """ Train on both cultures, test on Serbia """
    c1_m3_weights = __run_deep_net(c1_data_merged, 'm3') 

    print('---------- Completed Model 3 ----------')

    return c0_m3_weights, c1_m3_weights

def run_m4(c0_data, c1_data, c0_data_merged, c1_data_merged, c0_m3_weights, c1_m3_weights): 
    """ 
    Runs Model 4 - Joint Culture / SI (CultureNet)
    Train on both cultures, fine tune with culture A, test on culture A 

    PARAMETERS 
    c0_data: tuple of loaded data for culture 0 
    c1_data: tuple of loaded data for culture 1 
    c0_data_merged: tuple of merged data, with culture 0 as target 
    c1_data_merged: tuple of merged data, with culture 1 as target 
    c0_m3_weights: weights from model 3 for culture 0 
    c1_m3_weights: weights from model 3 for culture 1 
    """ 

    print('---------- Running Model 4 ----------')

    """ Train on both cultures, fine tune and test on Japan """

    if c0_m3_weights is None: 
        __run_deep_joint(c0_data, c1_data, c0_data_merged, 'm4_prelim', 'm4')
    else: 
        _ = __run_deep_net(c0_data, 'm4', trainable=[False,False,False,False,True], weights=c0_m3_weights)

    """ Train on both cultures, fine tune and test on Serbia """ 

    if c1_m3_weights is None: 
        __run_deep_joint(c1_data, c0_data, c1_data_merged, 'm4_prelim', 'm4')
    else:
        _ = __run_deep_net(c1_data, 'm4', trainable=[False,False,False,False,True], weights=c1_m3_weights)

    print('---------- Completed Model 4 ----------')

    return None

def run_m5(c0_data_targetRep, c1_data_targetRep): 
    """ 
    Runs Model 5 - Joint Culture / SD (GenNet) 
    Train and test on each culture, including 20% of target data 

    PARAMETERS 
    c0_data_targetRep: tuple of loaded data for culture 0 (includes 20% of target data)
    c1_data_targetRep: tuple of loaded data for culture 1 (includes 20% of target data)
    """ 

    print('---------- Running Model 5 ----------')

    """ Train and test on Japan """ 
    _ = __run_deep_net(c0_data_targetRep, 'm5')

    """ Train and test on Serbia """ 
    _ = __run_deep_net(c1_data_targetRep, 'm5')

    print('---------- Completed Model 5 ----------')

    return None

def run_m6(c0_data_targetOnly, c1_data_targetOnly): 
    """ 
    Runs Model 6 - Individual / SD 
    Train and test on each culture, using only 20% of target data 

    PARAMETERS 
    c0_data_targetOnly: tuple of loaded data for culture 0 (only 20% of target data)
    c1_data_targetOnly: tuple of loaded data for culture 1 (only 20% of target data)
    """ 

    print('---------- Running Model 6 ----------')

    """ Train and test on Japan """ 
    _ = __run_deep_net(c0_data_targetOnly, 'm6')

    """ Train and test on Serbia """ 
    _ = __run_deep_net(c1_data_targetOnly, 'm6')

    print('---------- Completed Model 6 ----------')

    return None

def run_prelim_m7(m7_joint_data, c0_data_All, c1_data_All):
    """
    Runs Preliminary Model 7 - Joint Culture / SD 
    Train on both cultures, fine tune with culture A 
    Note: Run before for loop over children 

    PARAMETERS 
    m7_joint_data: tuple of loaded data for cultures 0 and 1 
    c0_data_All: tuple of loaded data for culture 0 (all children in training, validation, test)
    c1_data_All: tuple of loaded data for culture 1 (all children in training, validation, test)
    """

    print('---------- Running Preliminary Model 7 ----------')

    # Part A: Joint Model (train & validate on 20% of all children) 
    prelim_weights = __run_deep_net(m7_joint_data, 'm7_joint_prelim')

    # Part B: Culture Specific Model (train & validate on 20% of all children)
    c0_m7_prelim_weights = __run_deep_net(c0_data_All, 'm7_prelim', trainable=[False,False,False,False,True], weights=prelim_weights)
    c1_m7_prelim_weights = __run_deep_net(c1_data_All, 'm7_prelim', trainable=[False,False,False,False,True], weights=prelim_weights)

    print('---------- Completed Preliminary Model 7 ----------')

    return c0_m7_prelim_weights, c1_m7_prelim_weights

def run_m7(c0_data_targetOnly, c1_data_targetOnly, c0_m7_prelim_weights, c1_m7_prelim_weights): 
    """ 
    Runs Model 7 - Joint Culture / SD 
    Train on both cultures (prelim), fine tune with culture A (prelim), fine tune with target data, test on culture A 
    Note: Run after preliminary model 

    PARAMETERS 
    c0_data_targetOnly: tuple of loaded data for culture 0 (only 20% of target data)
    c1_data_targetOnly: tuple of loaded data for culture 1 (only 20% of target data)
    c0_m7_prelim_weights: weights from preliminary model for culture 0 
    c1_m7_prelim_weights: weights from preliminary model for culture 1
    """ 

    print('---------- Running Model 7 ----------')

    # Part C: Child Specific Model 
    _ = __run_deep_net(c0_data_targetOnly, 'm7', trainable=[False,False,False,False,True], weights=c0_m7_prelim_weights)
    _ = __run_deep_net(c1_data_targetOnly, 'm7', trainable=[False,False,False,False,True], weights=c1_m7_prelim_weights)

    print('---------- Completed Model 7 ----------')

    return None

if __name__ == '__main__':
    pass 