import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
import os

train = pd.read_csv('./train.csv')
train = train.sort_values(['PID', 'UID']).reset_index(drop=True)

# add before ef
bef = []
PID = ''
mean_ef = 61
for index in range(len(train)):
    col = train.loc[index, :]
    if PID != col['PID']:
        bef.append(mean_ef)
        PID = col['PID']
        EF = col['EF']
    else:
        bef.append(EF)
        EF = col['EF']

train['BEF'] = bef
train.to_csv('./train.csv', index=False)
# split data
keep = []
PID = ''
for index in range(len(train)):
    col = train.loc[index, :]
    if PID != col['PID']:
        PID = col['PID']
    else:
        keep.append(index)

train = train.reset_index()
train = train[train['index'].isin(keep)]
train.drop('index', axis=1, inplace=True)

train = train.sample(frac=1).reset_index(drop=True)
train[:int(len(train)*0.2)].to_csv('./data/validation.csv', index=False)
train[int(len(train)*0.2):].to_csv('./data/train.csv', index=False)

# preprocessing test data
PID = ''
bdf = pd.DataFrame()
for index in range(len(train)):
    col = train.loc[index, :]
    uid = col['UID']
    if PID != col['PID']:
        PID = col['PID']
        bdf = pd.read_csv(f'ecg/{uid}.csv')
    else:
        df = pd.read_csv(f'ecg/{uid}.csv')
        pd.concat([df, bdf], axis=1).to_csv(f'./pytorch/data/ecg/{uid}.csv', index=False)
        bdf = df
train.drop_duplicates(subset=['PID'], keep='last')
test = pd.read_csv('./test.csv')
test_uid = test.merge(train.drop_duplicates(subset=['PID'], keep='last')[['PID', 'UID']], on='PID', how='left')
for uid_x, uid_y in zip(test_uid['UID_x'], test_uid['UID_y']):
    pd.concat([pd.read_csv(f'ecg/{uid_x}.csv'), pd.read_csv(f'ecg/{uid_y}.csv')], axis=1).to_csv(f'./data/ecg/{uid_x}.csv', index=False)
test = test.merge(train.drop_duplicates(subset=['PID'], keep='last')[['PID', 'EF']], on='PID', how='left')
test = test.rename({'EF': 'BEF'}, axis=1)
test.to_csv('./data/test.csv', index=False)

# normalize data
std_dict = {'leadI':Normalizer(), 'leadII':Normalizer(), 'leadIII':Normalizer(),
            'leadaVR':Normalizer(), 'leadaVL':Normalizer(), 'leadaVF':Normalizer(),
            'leadV1':Normalizer(), 'leadV2':Normalizer(), 'leadV3':Normalizer(),
            'leadV4':Normalizer(), 'leadV5':Normalizer(), 'leadV6':Normalizer()}

file_list = os.listdir('ecg')
for file in file_list:
    df = pd.read_csv(f'ecg/{file}')
    for col in df.columns:
        std_dict[col].fit(df[col].values.reshape(1, -1))
		
for file in file_list:
    df = pd.read_csv(f'ecg/{file}')
    for col in df.columns:
        df[col] = std_dict[col].transform(df[col].values.reshape(1, -1)).reshape(-1)
    df.to_csv(f'./pytorch/data/ecg/{file}', index=False)
