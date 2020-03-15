#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    labels: labels
    classes: number of classes
    """
    encoded = K.utils.to_categorical(labels, num_classes=classes)
    return encoded
