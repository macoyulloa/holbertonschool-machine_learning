#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Arg:
        network: neural model
        data: inputs
        labels: correct one-hot label
        verbose: determine if output should be printed
    Return: loss and acc of the model
    """
    return network.evaluate(data, labels, verbose=verbose)
