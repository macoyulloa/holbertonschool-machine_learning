#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


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
    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose, shuffle=shuffle)
    return history
