#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow as tf


def one_hot(labels, classes=None):
    """
    labels: labels
    classes: number of classes
    """
    encoded = tf.keras.utils.to_categorical(labels)
    return encoded
