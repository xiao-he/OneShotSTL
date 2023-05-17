import os
import json
import torch
import numpy as np
import pandas as pd

# Preprocessing code from Informer2020/utils/tools.py at https://github.com/zhouhaoyi/Informer2020
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    
# Preprocessing code from Informer2020/data/data_loader.py at https://github.com/zhouhaoyi/Informer2020
def read_data(name, features='S'):
    if name.startswith('ETTh'):
        df_raw = pd.read_csv('data/all_six_datasets/ETT-small/{}.csv'.format(name))
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24] 
        df_raw = df_raw[:border2s[2]]
    elif name.startswith('ETTm'):
        df_raw = pd.read_csv('data/all_six_datasets/ETT-small/{}.csv'.format(name))
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        df_raw = df_raw[:border2s[2]]
    else:
        if name == 'illness':
            df_raw = pd.read_csv('data/all_six_datasets/illness/national_illness.csv')
        else:
            df_raw = pd.read_csv('data/all_six_datasets/{}/{}.csv'.format(name, name))
        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test            
        border2s = [num_train, num_train+num_vali, len(df_raw)]
    if features == 'S':
        df_raw = df_raw['OT']
    else:
        cols_data = df_raw.columns[1:]
        df_raw = df_raw[cols_data]        
    return df_raw, border2s

if __name__ == '__main__':
    features = 'S'
    meta = pd.read_csv('data/all_six_datasets_meta.csv').to_numpy()
    for i in range(len(meta)):
        name = meta[i][0]
        period = meta[i][1]
        df_raw, border2s = read_data(name, features)
        train_test_split = border2s[1]
        scaler = StandardScaler()
        train_data = df_raw[:border2s[0]]
        scaler.fit(train_data.values)
        ts = scaler.transform(df_raw.values)
        data = {}
        data['ts'] = [float(x) for x in list(ts)]
        data['period'] = int(period)
        data['trainTestSplit'] = int(train_test_split)
        if name.startswith('ETT'):
            fn = 'data/all_six_datasets/ETT-small/{}_{}.json'.format(name, features)
        else:
            fn = 'data/all_six_datasets/{}/{}_{}.json'.format(name, name, features)
        with open(fn, "w") as outfile:
            json.dump(data, outfile)
        print(name, train_test_split, len(ts))