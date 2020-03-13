#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow as tf


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Arg:
        network: is the model to optimize
        data: numpy (m, nx)
        labels: one-hot code shape (m, classes)
        batch_size: size of the batch used foor mini-batch
        epochs: number of passes for data
        verbose: determines if output should be printed
        shuffle: determines whether to shuffle
    return: History object generated
    """
    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size)
    return history.history.keys()
