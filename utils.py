import numpy as np
import os
import pickle as pkl
from scipy.stats import pearsonr

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def load_data(data_ids, cul, data_proportion=[1,0,1,1]):
    """ 
    Loads data for a given country 

    PARAMETERS 
    data_ids: list of training, validation, and testing IDs 
    cul: country index 
    data_proportion: specifies proportions of data for training, validation, and testing; 
                     i.e. given [a,b,c,d], 
                     [:a*rows] = for training, [b*rows:c*rows] for validation, [(1-d)*rows:] for testing
    """
    # Creates array with all csv data for country X (specified by filter_str)
    # array columns: country ID, child ID, frame, feature array, label 
    full_raw = None 
    
    with open('./data/data_{}.pkl'.format(cul), 'rb') as f:
             full_raw = pkl.load(f)

    # Gets training data 
    country_train = None
    id_train = None 
    frame_train = None
    x_train = None
    y_train = None

    for train_id in data_ids[0]: # Loops over each training ID 
        train_data = full_raw[full_raw[:,1] == train_id] # Retrieves all rows where first column has train_ID 
        rows, _ = train_data.shape
        cutoff_train = int(rows*data_proportion[0]) # Computes cutoff value for training data, based on data_proportions input 

        if id_train is None: 
            country_train = train_data[:cutoff_train, :1]
            id_train = train_data[:cutoff_train, 1:2]
            frame_train = train_data[:cutoff_train, 2:3]
            x_train = train_data[:cutoff_train, 3:-1]
            y_train = train_data[:cutoff_train, -1:]
        else:
            country_train = np.vstack((country_train, train_data[:cutoff_train, :1]))
            id_train = np.vstack((id_train, train_data[:cutoff_train, 1:2]))
            frame_train = np.vstack((frame_train, train_data[:cutoff_train, 2:3]))
            x_train = np.vstack((x_train, train_data[:cutoff_train, 3:-1]))
            y_train = np.vstack((y_train, train_data[:cutoff_train, -1:]))

    # Gets validation data 
    country_val = None
    id_val = None 
    frame_val = None
    x_val = None
    y_val = None

    for val_id in data_ids[1]: # Loop over each validation ID 
        val_data = full_raw[full_raw[:,1] == val_id] # Retrieves all rows where first column has val_id 
        rows, _ = val_data.shape 
        cutoff_val_1 = int(rows*data_proportion[1]) # Computes lower cutoff value for validation data, based on data_proportions input 
        cutoff_val_2 = int(rows*data_proportion[2]) # Computes upper cutoff value for validation data, based on data_proportions input 

        if id_val is None: 
            country_val = val_data[cutoff_val_1:cutoff_val_2, :1]
            id_val = val_data[cutoff_val_1:cutoff_val_2, 1:2]
            frame_val = val_data[cutoff_val_1:cutoff_val_2, 2:3]
            x_val = val_data[cutoff_val_1:cutoff_val_2, 3:-1]
            y_val = val_data[cutoff_val_1:cutoff_val_2, -1:]
        else:
            country_val = np.vstack((country_val, val_data[cutoff_val_1:cutoff_val_2, :1]))
            id_val = np.vstack((id_val, val_data[cutoff_val_1:cutoff_val_2, 1:2]))
            frame_val = np.vstack((frame_val, val_data[cutoff_val_1:cutoff_val_2, 2:3]))
            x_val = np.vstack((x_val, val_data[cutoff_val_1:cutoff_val_2, 3:-1]))
            y_val = np.vstack((y_val, val_data[cutoff_val_1:cutoff_val_2, -1:]))

    # Gets testing data 
    country_test = None
    id_test = None 
    frame_test = None
    x_test = None
    y_test = None

    for test_id in data_ids[2]: # Loops over each testing ID 
        test_data = full_raw[full_raw[:,1] == test_id] # Retrieves all rows where first column has test_id 
        rows, _ = test_data.shape
        cutoff_test = int(rows*(1-data_proportion[3]))

        if id_test is None: 
            country_test = test_data[cutoff_test:, :1]
            id_test = test_data[cutoff_test:, 1:2]
            frame_test = test_data[cutoff_test:, 2:3]
            x_test = test_data[cutoff_test:, 3:-1]
            y_test = test_data[cutoff_test:, -1:]
        else:
            country_test = np.vstack((country_test, test_data[cutoff_test:, :1]))
            id_test = np.vstack((id_test, test_data[cutoff_test:, 1:2]))
            frame_test = np.vstack((frame_test, test_data[cutoff_test:, 2:3]))
            x_test = np.vstack((x_test, test_data[cutoff_test:, 3:-1]))
            y_test = np.vstack((y_test, test_data[cutoff_test:, -1:]))
    
    return id_train, id_val, id_test, x_train, x_val, x_test, y_train, y_val, y_test, country_train, country_val, country_test, frame_train, frame_val, frame_test

def leave_1_out_ids(IDs):
    """ Generates leave-1-out ID list """ 

    fold_list = []
    
    for i in range(len(IDs)):
        rolled = np.roll(IDs, i).tolist()
        fold_list.append([rolled[1:], rolled[1:], rolled[0:1]])
    
    return fold_list

def all_children_ids(IDs):
    """ Generates all children ID list """ 

    fold_list = []

    for i in range(len(IDs)):
        rolled = np.roll(IDs, i).tolist()
        fold_list.append([rolled[:], rolled[0:1], rolled[0:1]])

    return fold_list

def target_only_ids(IDs):
    """ Generates target-only ID list """ 

    return list(map(lambda x:[[x], [x], [x]], IDs))

def merge_data(cA_data, cB_data): 
    """ Merges data for models 3 and 4 """ 
    id_train = np.concatenate((cA_data[0], cB_data[0], cB_data[1], cB_data[2]), axis=0)
    id_val, id_test = cA_data[1], cA_data[2]

    x_train = np.concatenate((cA_data[3], cB_data[3], cB_data[4], cB_data[5]), axis=0)
    x_val, x_test = cA_data[4], cA_data[5]

    y_train = np.concatenate((cA_data[6], cB_data[6], cB_data[7], cB_data[8]), axis=0)
    y_val, y_test = cA_data[7], cA_data[8]

    culture_train = np.concatenate((cA_data[9], cB_data[9], cB_data[10], cB_data[11]), axis=0)
    culture_val, culture_test = cA_data[10], cA_data[11]

    frame_train = np.concatenate((cA_data[12], cB_data[12], cB_data[13], cB_data[14]), axis=0)
    frame_val, frame_test = cA_data[13], cA_data[14]

    prelim_data = (id_train, id_val, id_test, x_train, x_val, x_test, y_train, y_val, y_test, culture_train, culture_val, culture_test, frame_train, frame_val, frame_test)

    return prelim_data

def _process(y_hat, y_lab, fun):
    """
    Splits y_true and y_pred in lists
    Removes frames where labels are unknown (-1)
    Returns list of predictions
    """
    y1 = [x for x in y_hat.T]
    y2 = [x for x in y_lab.T]

    out = []
    for _, [_y1, _y2] in enumerate(zip(y1, y2)):
        idx = _y2!=-1
        _y1 = _y1[idx]
        _y2 = _y2[idx]
        if np.all(_y2==-1):
            out.append(np.nan)
        else:
            out.append(fun(_y1,_y2))
    return np.array(out)

def _icc(y_hat, y_lab, cas=3, typ=1):
    """ Computes intra-class correlation """
    def fun(y_hat,y_lab):
        y_hat = y_hat[None,:]
        y_lab = y_lab[None,:]

        Y = np.array((y_lab, y_hat))
        # number of targets
        n = Y.shape[2]
        # mean per target
        mpt = np.mean(Y, 0)
        # print mpt.eval()
        mpr = np.mean(Y, 2)
        # print mpr.eval()
        tm = np.mean(mpt, 1)
        # within target sum sqrs
        WSS = np.sum((Y[0]-mpt)**2 + (Y[1]-mpt)**2, 1)
        # within mean sqrs
        WMS = WSS/n
        # between rater sum sqrs
        RSS = np.sum((mpr - tm)**2, 0) * n
        # between rater mean sqrs
        RMS = RSS
        # between target sum sqrs
        TM = np.tile(tm, (y_hat.shape[1], 1)).T
        BSS = np.sum((mpt - TM)**2, 1) * 2
        # between targets mean squares
        BMS = BSS / (n - 1)
        # residual sum of squares
        ESS = WSS - RSS
        # residual mean sqrs
        EMS = ESS / (n - 1)

        if cas == 1:
            if typ == 1:
                res = (BMS - WMS) / (BMS + WMS)
            if typ == 2:
                res = (BMS - WMS) / BMS
        if cas == 2:
            if typ == 1:
                res = (BMS - EMS) / (BMS + EMS + 2 * (RMS - EMS) / n)
            if typ == 2:
                res = (BMS - EMS) / (BMS + (RMS - EMS) / n)
        if cas == 3:
            if typ == 1:
                res = (BMS - EMS) / (BMS + EMS)
            if typ == 2:
                res = (BMS - EMS) / BMS
        res = res[0]

        if np.isnan(res) or np.isinf(res):
            return 0
        else:
            return res
    return _process(y_hat, y_lab, fun)

def icc(y_hat, y_lab):
    return _icc(y_hat, y_lab)

def ccc(y_hat, y_lab):
    """ Computes concordance correlation coefficient """ 
    pred_mean = np.mean(y_hat)
    ref_mean = np.mean(y_lab)

    pred_var = np.var(y_hat)
    ref_var = np.var(y_lab)

    covariance = np.mean((y_hat - pred_mean) * (y_lab - ref_mean))

    return (2*covariance) / (pred_var+ref_var+np.power((pred_mean-ref_mean),2))

def mae(y_hat, y_lab):
    """ Computes mean absolute error """ 
    y_hat = np.asarray(y_hat)
    y_lab = np.asarray(y_lab)
    diff = np.subtract(y_hat, y_lab)
    abs_diff = np.fabs(diff)

    return float(sum(abs_diff)/len(y_lab))

def process_summary(REPORTS_FOLDER_DIR): 
    """ 
    Processes data 
    Saves file with average ICC per culture +/- STD, PCC per culture +/ STD, and CCC per culture +/ STD 
    """ 

    REPORTS_SUB_DIR = np.array([d for d in os.listdir(REPORTS_FOLDER_DIR) if os.path.isdir(os.path.join(REPORTS_FOLDER_DIR,d))])
    FINAL_REPORTS_SUB_DIR = [c for c in REPORTS_SUB_DIR if not(c.endswith('prelim'))]
    
    for report in FINAL_REPORTS_SUB_DIR: 
        CURRENT_REPORTS_SUB_DIR = os.path.join(REPORTS_FOLDER_DIR, report)
        ICC_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'icc_report.txt')
        ICC_FINAL_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'icc_final_report.csv')
        PCC_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'pcc_report.txt')
        PCC_FINAL_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'pcc_final_report.csv')
        CCC_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'ccc_report.txt')
        CCC_FINAL_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'ccc_final_report.csv')
        MAE_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'mae_report.txt')
        MAE_FINAL_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'mae_final_report.csv')

        FINAL_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'final_report.csv')

        icc_data = np.genfromtxt(ICC_DIR, delimiter=',')
        pcc_data = np.genfromtxt(PCC_DIR, delimiter=',')
        ccc_data = np.genfromtxt(CCC_DIR, delimiter=',')
        mae_data = np.genfromtxt(MAE_DIR, delimiter=',')
        
        cultures = np.unique(icc_data[:,0])
        culture_avg_icc = np.zeros((len(cultures),2))
        culture_avg_pcc = np.zeros((len(cultures),2))
        culture_avg_ccc = np.zeros((len(cultures),2))
        culture_avg_mae = np.zeros((len(cultures),2))
    
        for c in cultures: 
            ICC_CULTURE_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'c{}_icc_report.txt'.format(int(c)))
            PCC_CULTURE_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'c{}_pcc_report.txt'.format(int(c)))
            CCC_CULTURE_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'c{}_ccc_report.txt'.format(int(c)))
            MAE_CULTURE_DIR = os.path.join(CURRENT_REPORTS_SUB_DIR, 'c{}_mae_report.txt'.format(int(c)))
            
            culture_rows_icc = np.where(icc_data[:,0] == c)[0] # get row numbers for culture c
            culture_ids_icc = icc_data[culture_rows_icc,1] # get ID rows for culture c 
            unique_ids_icc = np.unique(culture_ids_icc) # get unique IDs for culture c 
            
            culture_rows_pcc = np.where(pcc_data[:,0] == c)[0] # get row numbers for culture c
            culture_ids_pcc = pcc_data[culture_rows_pcc,1] # get ID rows for culture c 
            unique_ids_pcc = np.unique(culture_ids_pcc) # get unique IDs for culture c 
            
            culture_rows_ccc = np.where(ccc_data[:,0] == c)[0] # get row numbers for culture c
            culture_ids_ccc = ccc_data[culture_rows_ccc,1] # get ID rows for culture c 
            unique_ids_ccc = np.unique(culture_ids_ccc) # get unique IDs for culture c 

            culture_rows_mae = np.where(mae_data[:,0] == c)[0] # get row numbers for culture c
            culture_ids_mae = ccc_data[culture_rows_mae,1] # get ID rows for culture c 
            unique_ids_mae = np.unique(culture_ids_mae) # get unique IDs for culture c 

            assert np.array_equal(unique_ids_icc, unique_ids_pcc)
            assert np.array_equal(unique_ids_icc, unique_ids_ccc)
            assert np.array_equal(unique_ids_icc, unique_ids_mae)

            culture_icc = None 
            culture_pcc = None 
            culture_ccc = None 
            culture_mae = None 
            
            for u in unique_ids_icc: 
                all_id_rows_icc = np.where(icc_data[:,1] == u)[0] 
                id_rows_icc = np.intersect1d(all_id_rows_icc, culture_rows_icc) # get ID rows for child u 
                
                all_id_rows_pcc = np.where(pcc_data[:,1] == u)[0] 
                id_rows_pcc = np.intersect1d(all_id_rows_pcc, culture_rows_pcc) # get ID rows for child u         
                
                all_id_rows_ccc = np.where(ccc_data[:,1] == u)[0] 
                id_rows_ccc = np.intersect1d(all_id_rows_ccc, culture_rows_ccc) # get ID rows for child u     

                all_id_rows_mae = np.where(mae_data[:,1] == u)[0] 
                id_rows_mae = np.intersect1d(all_id_rows_mae, culture_rows_mae) # get ID rows for child u     

                id_icc = icc_data[id_rows_icc, 2] # get ICC data for child u 
                avg_icc = np.mean(id_icc) # get average ICC for child u 

                id_pcc = pcc_data[id_rows_pcc, 2] # get PCC data for child u 
                avg_pcc = np.mean(id_pcc) # get average PCC for child u 

                id_ccc = ccc_data[id_rows_ccc, 2] # get CCC data for child u 
                avg_ccc = np.mean(id_ccc) # get average CCC for child u 

                id_mae = mae_data[id_rows_mae, 2] # get MAE data for child u 
                avg_mae = np.mean(id_mae) # get average MAE for child u 

                culture_icc = np.array([[u, avg_icc]]) if culture_icc is None else np.vstack((culture_icc, np.array([[u, avg_icc]])))
                culture_pcc = np.array([[u, avg_pcc]]) if culture_pcc is None else np.vstack((culture_pcc, np.array([[u, avg_pcc]])))
                culture_ccc = np.array([[u, avg_ccc]]) if culture_ccc is None else np.vstack((culture_ccc, np.array([[u, avg_ccc]])))
                culture_mae = np.array([[u, avg_mae]]) if culture_mae is None else np.vstack((culture_mae, np.array([[u, avg_mae]])))
                                
            np.savetxt(ICC_CULTURE_DIR, culture_icc)
            np.savetxt(PCC_CULTURE_DIR, culture_pcc)
            np.savetxt(CCC_CULTURE_DIR, culture_ccc)
            np.savetxt(MAE_CULTURE_DIR, culture_mae)
            
            culture_avg_icc[cultures.tolist().index(c), 0] = np.mean(culture_icc, axis=0)[1]
            culture_avg_icc[cultures.tolist().index(c), 1] = np.std(culture_icc[:,1])
            
            culture_avg_pcc[cultures.tolist().index(c), 0] = np.mean(culture_pcc, axis=0)[1]
            culture_avg_pcc[cultures.tolist().index(c), 1] = np.std(culture_pcc[:,1])

            culture_avg_ccc[cultures.tolist().index(c), 0] = np.mean(culture_ccc, axis=0)[1]
            culture_avg_ccc[cultures.tolist().index(c), 1] = np.std(culture_ccc[:,1])

            culture_avg_mae[cultures.tolist().index(c), 0] = np.mean(culture_mae, axis=0)[1]
            culture_avg_mae[cultures.tolist().index(c), 1] = np.std(culture_mae[:,1])

        np.savetxt(ICC_FINAL_DIR, culture_avg_icc, delimiter=',')
        np.savetxt(PCC_FINAL_DIR, culture_avg_pcc, delimiter=',') 
        np.savetxt(CCC_FINAL_DIR, culture_avg_ccc, delimiter=',') 
        np.savetxt(MAE_FINAL_DIR, culture_avg_mae, delimiter=',') 

        np.savetxt(FINAL_DIR, np.hstack((culture_avg_icc, culture_avg_pcc, culture_avg_ccc, culture_avg_mae)), delimiter=',')

    return None 

if __name__ == '__main__':
    pass