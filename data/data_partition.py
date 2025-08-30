import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ===========> Chapman DataSet preprocessing  <===========

# ===================================== normalization code from data_process.py
def preprocess_signals(X_train, X_val, X_test):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    return apply_standardizer(X_train, ss), apply_standardizer(X_val, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


# =====================================

sample_length = 1000

dir_data = '../data/ECGDataDenoised/'
save_filename = '../data/records100_denoise.pth'
diagnostics_info = pd.read_csv('../data/Diagnostics.csv')  # First use the tool to convert to a UTF-8 encoded CSV file

label2index = {
    'AFIB': 1, 'AF': 1,
    'SVT': 2, 'AT': 2, 'SAAWR': 2, 'ST': 2, 'AVNRT': 2, 'AVRT': 2,
    'SB': 3,
    'SR': 4, 'SA': 4
}

# Each section is balanced between men and women
data_male = []
label_male = []
data_female = []
label_female = []

start_time = time.time()

for index, row in diagnostics_info.iterrows():
    filename = row['FileName']
    try:
        ecg_data = pd.read_csv(dir_data + filename + '.csv').to_numpy().tolist()

        ecg_data = resample(ecg_data, sample_length, axis=0)

        rhythm = row['Rhythm']
        gender = row['Gender']

        if gender == 'MALE':
            data_male.append(ecg_data)
            label_male.append(label2index[rhythm])
        else:
            data_female.append(ecg_data)
            label_female.append(label2index[rhythm])

    except FileNotFoundError:
        print(FileNotFoundError)

data_male = np.array(data_male)
label_male = np.array(label_male)
data_female = np.array(data_female)
label_female = np.array(label_female)

data_male = torch.tensor(data_male).transpose(-1, -2)
label_male = torch.tensor(label_male)
data_female = torch.tensor(data_female).transpose(-1, -2)
label_female = torch.tensor(label_female)

# ===================================== delete samples with none values
# check NaN
nan_mask = torch.isnan(data_male)
# NaN indices
nan_indices = torch.nonzero(nan_mask)
idx = torch.unique(nan_indices.H[0])
# delete NaN
train_idx = torch.tensor([i for i in range(len(data_male)) if i not in idx])
data_male = data_male[train_idx].transpose(-1, -2).numpy()
label_male = label_male[train_idx].numpy()

nan_mask = torch.isnan(data_female)
nan_indices = torch.nonzero(nan_mask)
idx = torch.unique(nan_indices.H[0])
test_idx = torch.tensor([i for i in range(len(data_female)) if i not in idx])
data_female = data_female[test_idx].transpose(-1, -2).numpy()
label_female = label_female[test_idx].numpy()
# =====================================

# Train:(Val+Test) = 8:2
X_train1, X_val_test1, y_train1, y_val_test1 = train_test_split(data_male, label_male, test_size=0.2, random_state=10,
                                                                stratify=label_male)
X_train2, X_val_test2, y_train2, y_val_test2 = train_test_split(data_female, label_female, test_size=0.2,
                                                                random_state=10, stratify=label_female)

# Val:Test = 1:1
X_val1, X_test1, y_val1, y_test1 = train_test_split(X_val_test1, y_val_test1, test_size=0.5, random_state=10,
                                                    stratify=y_val_test1)
X_val2, X_test2, y_val2, y_test2 = train_test_split(X_val_test2, y_val_test2, test_size=0.5, random_state=10,
                                                    stratify=y_val_test2)

X_train = np.vstack((X_train1, X_train2))
y_train = np.hstack((y_train1, y_train2))
X_val = np.vstack((X_val1, X_val2))
y_val = np.hstack((y_val1, y_val2))
X_test = np.vstack((X_test1, X_test2))
y_test = np.hstack((y_test1, y_test2))

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_val = encoder.fit_transform(y_val.reshape(-1, 1))
y_test = encoder.fit_transform(y_test.reshape(-1, 1))

X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

data_dict = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}

# save
torch.save(data_dict, save_filename)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"time: {elapsed_time}")

print('finish')
