#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    network: model to save
    filename: path of the file
    """
    json_model = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_model)
    return None


def load_config(filename):
    """
    filename: path where the model should be loaded
    """
    with open(filename, 'r') as json_file:
        json_model = K.models.model_from_json(json_file.read())
    return json_model
