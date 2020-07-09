#!/usr/bin/env python3
""" preprocessing the database: forecast analysis
    of the Bitcoins """

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def split_datasets(dataset, target, start_index, end_index, history_size,
                   target_size, step, single_step=False):
    """ Time window of 24 hours

    Arg:
        - dataset: np.ndarray shape (num_datapoints, num_features)
                    with all features to do the forecast
        - start_index: starting point in the dataset
        - end_index: ending point in the dataset
        - history_size: time windows, for this case is an int 24
        - target_size: int label that needs to be predicted.

    Returns: datasets with the windows 24 hours for the model to train on
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        # Reshape data from (history_size,) to (history_size, 1)
        if (dataset.shape[1] == 1):
            data.append(np.reshape(dataset[indices], (history_size, 1)))
        else:
            data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


def data_preprocessing(path_file):
    """ Preprocess the data from the dataset.
        Here are some things to consider:

        Arg:
            - path_file: where is the csv is

        Returns:
            - dataset: np.ndarray of shape (, 6) represents the number of data
                per feature of the table.
            - features: containing the features of the datframe
            - x_train: training dataset
            - y_train: labels for the training process
            - x_val: datapoints of the validation dataset
            - y_val: labels for the validation process
    """
    df = pd.read_csv(path_file)
    # removing the NaN values, invalid values
    df.rename(columns={'Volume_(Currency)': 'VolumeUSD',
                       'Volume_(BTC)': 'VolumeBTC'},
              inplace=True)
    df = df.dropna()
    # getting the decoded date
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], infer_datetime_format=True, unit='s')

    # convert the time serie per each hour
    df = df.set_index('Timestamp')
    # selecting the range from the dataset of the data to work with

    df['VolumeUSD'] = df.resample('H').sum()
    df['Low'] = df.resample('H').min()
    df['High'] = df.resample('H').max()
    df['Open'] = df.resample('H').mean()
    df['Close'] = df.resample('H').mean()
    df['Weighted_Price'] = df.resample('H').mean()
    df = df.dropna()
    df = df.iloc[-366:]

    # selecting the features to take into account
    features_consider = ['Open', 'Low', 'High',
                         'Close', 'VolumeUSD',
                         'Weighted_Price']
    features = df[features_consider]
    # print(features.index)
    # features.index = lastYear_data['Timestamp']

    TRAIN_SPLIT = 269

    # tf.random.set_seed(13)
    # standardize the dataset using the mean and standard deviation
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset - data_mean) / data_std

    past_history = 24
    future_target = 1
    STEP = 1
    x_train, y_train = split_datasets(dataset, dataset[:, 3],
                                      0, TRAIN_SPLIT,
                                      past_history,
                                      future_target, STEP,
                                      single_step=True)
    x_val, y_val = split_datasets(dataset, dataset[:, 3],
                                  TRAIN_SPLIT, None,
                                  past_history,
                                  future_target, STEP,
                                  single_step=True)

    return dataset, features, x_train, y_train, x_val, y_val


if __name__ == "__main__":
    """
    """
    file_path = './bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    dataset, features, x_train, y_train, x_val, y_val = data_preprocessing(
        file_path)
    print('Single window of past history : {}'.format(x_train[0].shape))
    print(dataset.shape)
    print(dataset)
    print("train: {}, y_train: {}, x_val: {}, y_val: {}".format(
        x_train.shape, y_train.shape, x_val.shape, y_val.shape))
    # ploting the features of the data
    print(features.values)
    print(features.index)
    features.Weighted_Price.plot(subplots=True)
    plt.show()
