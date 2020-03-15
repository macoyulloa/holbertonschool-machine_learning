#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow as tf


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
    network = tf.keras.models.load_model(filename)
    return network
