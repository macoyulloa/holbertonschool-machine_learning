#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    network: model to save
    filename: path of the file
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    filename: path where the model should be loaded
    """
    network = K.models.load_model(filename)
    return network
