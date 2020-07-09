#!/usr/bin/env python3
""" forecasting the database: forecast analysis
    of the Bitcoins """


import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
data_preprocessing = __import__('preprocess_data').data_preprocessing


def create_time_steps(length):
    """
    """
    return list(range(-length, 0))


def plot_train_history(history, title):
    """
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def show_plot(plot_data, delta, title):
    """
    """
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(
            ), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def forecasting(x_train, y_train, x_val, y_val, BUFFER_SIZE,
                BATCH_SIZE, EPOCHS, EVAL_INTERVAL):
    """ Forecasting model of the BTC price

        Arg:

        Returns:

    """
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32,
                                               input_shape=x_train.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(
        optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    single_step_history = single_step_model.fit(train_data, epochs=EPOCHS,
                                                steps_per_epoch=EVAL_INTERVAL,
                                                validation_data=val_data,
                                                validation_steps=50)
    plot_train_history(single_step_history,
                       'Single Step Training and validation loss')

    # Predict a single step future
    for x, y in val_data.take(3):
        plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                          single_step_model.predict(x)[0]], 12,
                         'Single Step Prediction')
    plot.show()


if __name__ == "__main__":
    """ starting point of the model
    """
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000
    EVAL_INTERVAL = 500
    EPOCHS = 20
    file_path = './bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    dataset, features, x_train, y_train, x_val, y_val = data_preprocessing(
        file_path)
    print(dataset.shape)
    print(dataset)
    print("train: {}, y_train: {}, x_val: {}, y_val: {}".format(
        x_train.shape, y_train.shape, x_val.shape, y_val.shape))
    # ploting the features of the data
    # print(features.values)
    # print(features.index)
    features.Weighted_Price.plot(subplots=True)
    plt.show()
    forecasting(x_train, y_train, x_val, y_val, BUFFER_SIZE,
                BATCH_SIZE, EPOCHS, EVAL_INTERVAL)
