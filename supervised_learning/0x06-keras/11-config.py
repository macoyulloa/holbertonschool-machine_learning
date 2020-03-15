#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow as tf


def save_config(network, filename):
    """
    network: model to save
    filename: path of the file
    """
    filename = network.to_json()
    return None


def load_config(filename):
    """
    filename: path where the model should be loaded
    """
    return tf.keras.models.model_from_json(filename)
