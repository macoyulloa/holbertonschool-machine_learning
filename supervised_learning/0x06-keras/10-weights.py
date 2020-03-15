#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    network: model to save
    filename: path of the file
    save_format: weights format saved
    """
    network.save_weights(filename)
    return None


def load_weights(network, filename):
    """
    network: model to wigh the wieghts should be loaded
    filename: path where the model should be loaded
    """
    network.load_weights(filename)
    return None
