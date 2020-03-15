#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Arg:
        network: neural model
        data: inputs
        verbose: determine if output should be printed
    Return: prediction for the data
    """
    return network.predict(data, verbose=verbose)
