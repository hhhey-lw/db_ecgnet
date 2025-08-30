import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .data_process import load_dataset, compute_label_aggregations, select_data, data_slice, preprocess_signals, chapman_dataset
from scipy.signal import butter, filtfilt
import os


class Denoise:
    def __init__(self, sampling_rate=100):
        # Butterworth high-pass filter
        cutoff = 0.5
        order = 5
        nyquist = 0.5 * sampling_rate
        cutoff /= nyquist
        self.b, self.a = butter(order, cutoff, btype='highpass')

    def removeBaselineDrift(self, ecg_data):
        return filtfilt(self.b, self.a, ecg_data)


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        super(ECGDataset, self).__init__()
        self.data = signals
        self.label = labels
        self.num_classes = self.label.shape[1]

        self.cls_num_list = np.sum(self.label, axis=0)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]

        x = x.transpose()

        x = torch.tensor(x.copy(), dtype=torch.float)

        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()
        return x, y

    def __len__(self):
        return len(self.data)


class DownLoadECGData:
    """
        All experiments data    - 1-10 fold
    """
    def __init__(self, task, datafolder, sampling_frequency=100, min_samples=0,
                 val_fold=9, test_fold=10):
        self.min_samples = min_samples
        self.task = task
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency
        self.sampling_time = 10  # 10s

    # Train:Validate:Test -  8:1:1
    def preprocess_data(self):
        print(f'Validate Set: {self.val_fold}-Fold, Test Set: {self.test_fold}-Fold.')

        if self.datafolder.split("/")[-2] != 'ptbxl':
            file_name = f'records{self.sampling_frequency}_denoise.pth'
        else:
            file_name = f'records{self.sampling_frequency}_denoise_{self.task}.pth'
        file_path = os.path.join(self.datafolder, file_name)

        if os.path.exists(file_path):
            if self.datafolder.split("/")[-2] == 'ptbxl':
                loaded_data = joblib.load(file_path)
            else:
                loaded_data = torch.load(file_path)
            X_train = loaded_data['X_train']
            y_train = loaded_data['y_train']
            X_val = loaded_data['X_val']
            y_val = loaded_data['y_val']
            X_test = loaded_data['X_test']
            y_test = loaded_data['y_test']
        else:
            # Load data
            data, raw_labels = load_dataset(self.datafolder, self.sampling_frequency)
            # Preprocess label data
            labels = compute_label_aggregations(raw_labels, self.datafolder, self.task)

            # Select relevant data and convert to one-hot
            data, labels, Y, _ = select_data(data, labels, self.task, self.min_samples)

            if self.datafolder.split("/")[-2] == 'CPSC':
                data = data_slice(data, self.sampling_frequency, self.sampling_time)

            # 10th fold for testing (9th for now)
            X_test = data[labels.strat_fold == self.test_fold]
            y_test = Y[labels.strat_fold == self.test_fold]
            # 9th fold for validation (8th for now)
            X_val = data[labels.strat_fold == self.val_fold]
            y_val = Y[labels.strat_fold == self.val_fold]
            # rest for training
            X_train = data[(labels.strat_fold != self.test_fold) & (labels.strat_fold != self.val_fold)]
            y_train = Y[(labels.strat_fold != self.test_fold) & (labels.strat_fold != self.val_fold)]

            # remove baseline drift
            print('remove baseline drift')
            denoise = Denoise()
            X_train = np.swapaxes(X_train, -1, -2)
            X_val = np.swapaxes(X_val, -1, -2)
            X_test = np.swapaxes(X_test, -1, -2)
            for i, ecg in enumerate(X_train):
                X_train[i] = denoise.removeBaselineDrift(ecg.tolist())
            for i, ecg in enumerate(X_val):
                X_val[i] = denoise.removeBaselineDrift(ecg.tolist())
            for i, ecg in enumerate(X_test):
                X_test[i] = denoise.removeBaselineDrift(ecg.tolist())
            # keep origin format
            X_train = np.swapaxes(X_train, -1, -2)
            X_val = np.swapaxes(X_val, -1, -2)
            X_test = np.swapaxes(X_test, -1, -2)

            # Preprocess signal data  => normalization
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
            if self.datafolder.split("/")[-2] == 'ptbxl':
                joblib.dump(data_dict, file_path)
            else:
                torch.save(data_dict, file_path)

        return X_train, y_train, X_val, y_val, X_test, y_test


# Train:Validate:Test  -  8:1:1
def load_datasets(datafolder=None, experiment=None, batch_size=32, sampling_frequency=100):
    name_dataset = datafolder.split("/")[-2]
    if name_dataset == 'ptbxl':
        print('Loading dataset PTB-XL ... task is ' + experiment)
        # rhythm: ['AFIB' 'AFLT' 'BIGU' 'PACE' 'PSVT' 'SARRH' 'SBRAD' 'SR' 'STACH' 'SVARR' 'SVTAC' 'TRIGU']
        # experiments = ['all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm' ]
        ded = DownLoadECGData(experiment, datafolder, sampling_frequency)
        X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()
    elif name_dataset == 'CPSC':
        print('Loading dataset CPSC ...')
        ded = DownLoadECGData('all', datafolder, sampling_frequency)
        X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()
    else:
        print('Loading dataset chapman ...')
        print('Validate: 9fold, Test: 10fold')
        print('AFIB, GSVT, SB, SR')
        X_train, y_train, X_val, y_val, X_test, y_test = chapman_dataset(datafolder)

    ds_train = ECGDataset(X_train, y_train)
    ds_val = ECGDataset(X_val, y_val)
    ds_test = ECGDataset(X_test, y_test)

    num_classes = ds_train.num_classes
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, num_classes