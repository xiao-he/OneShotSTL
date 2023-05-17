import os
import json
from os import walk
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf
np.seterr(divide='ignore', invalid='ignore')

def my_find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    auto_corr = acf(data, nlags=len(data)//2, fft=True)
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        return local_max[max_local_max]
    except:
        return -1

if __name__ == '__main__':
    filepath = 'data/TSB-UAD-Public/'
    all_folders = []
    for (dirpath, dirnames, filenames) in walk(filepath):
        all_folders.extend(dirnames)
        break    
    for folder in all_folders:
        if folder != 'KDD21':
            all_fn = []
            for (dirpath, dirnames, filenames) in walk("{}/{}".format(filepath, folder)):
                all_fn.extend(filenames)
                break
            skip_cases = 0
            eval_cases = 0
            for fn in all_fn:
                if fn.endswith('.out'):
                    if folder in ['IOPS', 'NASA-MSL', 'NASA-SMAP']:
                        if 'test' in fn:
                            test_df = pd.read_csv('%s/%s/%s' % (filepath, folder, fn), header=None).to_numpy()
                            test_data = test_df[:,0].astype(float)
                            test_label = test_df[:,1]
                            train_fn = fn.replace('test', 'train')                            
                            train_df = pd.read_csv('%s/%s/%s' % (filepath, folder, train_fn), header=None).to_numpy()
                            train_data = train_df[:,0].astype(float)
                            train_label = train_df[:,1]
                            trainTestSplit = len(train_data)
                            T = my_find_length(train_data)
                            if T < 0 or np.sum(test_label) <= 0:
                                # print('Skip %s/%s/%s.meta' % (filepath, folder, fn))
                                skip_cases += 1
                            else:
                                eval_cases += 1     
                                data = {}
                                data['period'] = int(T)
                                data['trainTestSplit'] = int(trainTestSplit)
                                data['ts'] = [float(x) for x in list(np.concatenate([train_data, test_data]))]
                                data['label'] = [int(x) for x in list(np.concatenate([train_label, test_label]))]
                                with open('%s/%s/%s.json' % (filepath, folder, fn), "w") as outfile:
                                    json.dump(data, outfile)
                                # with open('%s/%s/%s.meta' % (filepath, folder, fn), 'w') as the_file:
                                    # the_file.write('{},{}\n'.format(T, trainTestSplit))     
                                # data = np.concatenate([train_data, test_data])
                                # label = np.concatenate([train_label, test_label])
                                # csv_data = np.zeros([data.shape[0], 2])
                                # csv_data[:, 0] = data
                                # csv_data[:, 1] = label
                                # np.savetxt('%s/%s/%s.csv' % (filepath, folder, fn), csv_data)                                      
                    else:
                        df = pd.read_csv('%s/%s/%s' % (filepath, folder, fn), header=None).dropna().to_numpy()
                        data = df[:,0].astype(float)
                        label = df[:,1]
                        trainTestSplit = min(3000, len(data)//10)
                        T = my_find_length(data[:trainTestSplit])
                        if T < 0:
                            # print('Skip %s/%s/%s.meta' % (filepath, folder, fn))
                            skip_cases += 1
                        else:                        
                            if trainTestSplit < T * 5:
                                trainTestSplit = T * 5
                            if np.sum(label[trainTestSplit:]) <= 0:
                                # print('Skip %s/%s/%s.meta' % (filepath, folder, fn))
                                skip_cases += 1
                            else:
                                eval_cases += 1
                                fdata = {}
                                fdata['period'] = int(T)
                                fdata['trainTestSplit'] = int(trainTestSplit)
                                fdata['ts'] = [float(x) for x in list(data)]
                                fdata['label'] = [int(x) for x in list(label)]
                                if folder == "OPPORTUNITY" and "fn" == "S4-ADL5.test.csv@55.out":
                                    import IPython as ip
                                    ip.embed()
                                with open('%s/%s/%s.json' % (filepath, folder, fn), "w") as outfile:
                                    json.dump(fdata, outfile)
                                # with open('%s/%s/%s.meta' % (filepath, folder, fn), 'w') as the_file:
                                #     the_file.write('{},{}\n'.format(T, trainTestSplit))
                                # csv_data = np.zeros([data.shape[0], 2])
                                # csv_data[:, 0] = data
                                # csv_data[:, 1] = label
                                # np.savetxt('%s/%s/%s.csv' % (filepath, folder, fn), csv_data)                                
            print('%s, eval number of ts:%s, skip number of ts:%s' % (folder, eval_cases, skip_cases))


    
