from utils import icc, ccc, mae 
from scipy.stats import pearsonr as pcc 

from keras.models import Model
from keras.layers import Dense, Input 
from keras.callbacks import EarlyStopping 
import os
import tensorflow as tf
import numpy as np

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

        # Generate report of summary statistics
        cultures = np.unique(country_test)
        for c in cultures:
            culture_rows = np.where(country_test == c)[0] # get row numbers for culture c 
            culture_ids = id_test[culture_rows] # get ID rows for culture c 
            unique_ids = np.unique(culture_ids) # get unique IDs for culture c 

            for u in unique_ids: 
                all_id_rows = np.where(id_test == u)[0]
                id_rows = np.intersect1d(all_id_rows, culture_rows) # get ID rows for child u 

                id_icc = icc(results[id_rows], y_test[id_rows])[0] # compute ICC for child u 
                id_pcc = pcc(results[id_rows], y_test[id_rows])[0][0] # compute PCC for child u 
                id_ccc = ccc(results[id_rows], y_test[id_rows]) # compute CCC for child u 
                id_mae = mae(results[id_rows], y_test[id_rows]) # compute MAE for child u 

                icc_entry = '{},{},{}\n'.format(c, u, id_icc)
                pcc_entry = '{},{},{}\n'.format(c, u, id_pcc)
                ccc_entry = '{},{},{}\n'.format(c, u, id_ccc)
                mae_entry = '{},{},{}\n'.format(c, u, id_mae)
                
                with open('Reports/{}/icc_report.txt'.format(report_name), 'a') as f:
                    f.write(icc_entry)

                with open('Reports/{}/pcc_report.txt'.format(report_name), 'a') as f:
                    f.write(pcc_entry)

                with open('Reports/{}/ccc_report.txt'.format(report_name), 'a') as f:
                    f.write(ccc_entry)

                with open('Reports/{}/mae_report.txt'.format(report_name), 'a') as f:
                    f.write(mae_entry)

        return results 

if __name__ == '__main__':
    pass