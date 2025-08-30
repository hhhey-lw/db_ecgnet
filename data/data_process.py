import ast
import os
import pickle

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from tqdm import tqdm


# DATA PROCESSING
def load_dataset(path, sampling_rate):
    if path.split('/')[-2] == 'ptbxl':
        # load and convert annotation data
        Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    elif path.split('/')[-2] == 'CPSC':
        # load and convert annotation data
        Y = pd.read_csv(path + 'cpsc_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_cpsc(Y, sampling_rate, path)

    return X, Y


def load_raw_data_cpsc(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path + 'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/' + str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data], dtype=object)
            pickle.dump(data, open(path + 'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 250:
        if os.path.exists(path + 'raw250.npy'):
            data = np.load(path + 'raw250.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records250/' + str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data], dtype=object)
            pickle.dump(data, open(path + 'raw250.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path + 'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records500/' + str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data], dtype=object)
            pickle.dump(data, open(path + 'raw500.npy', 'wb'), protocol=4)
    return data


def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path + 'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_lr, desc="loading dataset")]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path + 'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_hr, desc="loading dataset")]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw500.npy', 'wb'), protocol=4)
    return data


def compute_label_aggregations(df, folder, ctype):
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))
    aggregation_df = None
    if ctype != 'all' and folder != '../data/CPSC/':
        aggregation_df = pd.read_csv(folder + 'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


def select_data(XX, YY, ctype, min_samples):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)

    elif ctype == 'all':
        # filter
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    print(mlb.classes_)
    return X, Y, y, mlb


def preprocess_signals(X_train, X_validation, X_test):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


# unified data length
def data_slice(data, sampling_frequency, sampling_time):
    data_len = sampling_frequency * sampling_time
    data_process = []
    for dat in data:
        if dat.shape[0] < data_len:
            dat = np.pad(dat, ((0, data_len - dat.shape[0]), (0, 0)), 'constant', constant_values=0)
            # dat = resample(dat, data_len, axis=0)
        elif dat.shape[0] > data_len:
            dat = dat[:data_len, :]
            # dat = resample(dat, 1000, axis=0)
        if dat.shape[1] != 12:
            dat = dat[:, 0:12]

        data_process.append(dat)
    return np.array(data_process)


# chapman Dataset
def chapman_dataset(root='../data/chapman/'):
    fn = 'records100_denoise.pth'
    loaded_data = torch.load(root + fn)
    X_train = loaded_data['X_train']
    y_train = loaded_data['y_train']
    X_val = loaded_data['X_val']
    y_val = loaded_data['y_val']
    X_test = loaded_data['X_test']
    y_test = loaded_data['y_test']

    return X_train, y_train, X_val, y_val, X_test, y_test