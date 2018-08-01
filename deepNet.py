from utils import icc

from keras.models import Model
from keras.layers import Dense, Input, concatenate, BatchNormalization, Activation, Dropout
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, CSVLogger
from keras import backend as K
import keras
import os
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

class deepNet(Model):
    def __init__(self, input_dim, trainable=[True,True,True,True,True], lsize0=500, lsize1=200, lsize2=100, lsize3=50, epochs=20, batch_size=128, **kwargs):
        self.input_dim = input_dim
        self.trainable = trainable
        self.lsize0 = lsize0
        self.lsize1 = lsize1
        self.lsize2 = lsize2
        self.lsize3 = lsize3 
        self.epochs = epochs 
        self.batch_size = batch_size

        inp = Input(shape=(input_dim,))
        l1 = Dense(lsize0, activation='relu', trainable=trainable[0])(inp)
        l2 = Dense(lsize1, activation='relu', trainable=trainable[1])(l1)
        l3 = Dense(lsize2, activation='relu', trainable=trainable[2])(l2)
        l4 = Dense(lsize3, activation='relu', trainable=trainable[3])(l3)
        l5 = Dense(1, activation='linear', trainable=trainable[4])(l4)        

        super(deepNet, self).__init__(inputs=[inp], outputs=[l5]) 

    def train_model(self, x_train, y_train, x_val, y_val, dtype):
        self.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mae'])
        stopper = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, mode='auto')
        self.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, callbacks=[stopper], validation_data=(x_val,y_val))
        self.save_weights('Weights/{}_weights.h5'.format(dtype)) 

    def make_report(self, report_name, id_test, x_test, y_test, country_test, frame_test):
        """ Runs evaluate on the provided data and generates a detailed error report """
        if not os.path.exists('Reports/' + report_name):
            os.mkdir('Reports/' + report_name)
        results = self.predict(x_test)
        # Get ICC measures
        icc_scores = icc(results, y_test)

        # Generate detailied evaluation report
        header = 'Country,Child,Frame'
        for output_layer in self.get_config()['output_layers']:
            header += ',{}_Actual'.format(output_layer[0])
        for output_layer in self.get_config()['output_layers']:
            header += ',{}_Prediction'.format(output_layer[0]) 
        header += '\n'

        with open('Reports/{}/evaluation_report.txt'.format(report_name), 'a') as f:
            if os.stat('Reports/{}/evaluation_report.txt'.format(report_name)).st_size == 0:
                f.write(header)
            for row in range(len(results)):
                entry = ','.join([str(i) for i in country_test[row]]) + ','
                entry += ','.join([str(i) for i in id_test[row]]) + ','
                entry += ','.join([str(i) for i in frame_test[row]]) + ','
                entry += ','.join([str(i) for i in y_test[row]]) + ','
                entry += ','.join([str(i) for i in results[row]]) + '\n'
                f.write(entry)

        results = self.evaluate(x_test, y_test)

        # Generate report of summary statistics
        # with open('Reports/{}/c{}_id{}_icc_report.txt'.format(report_name, ), 'a') as f:
        with open('Reports/{}/icc_report.txt'.format(report_name), 'a') as f:
            # for metric_name, value in zip(self.metrics_names, results):
            #     f.write('{}: {}\n'.format(metric_name, value))
            f.write('{}\n'.format(icc_scores))

        return results
