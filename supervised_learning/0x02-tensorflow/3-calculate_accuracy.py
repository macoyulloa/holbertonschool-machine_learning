#!/usr/bin/env python3
"""accurancy of a prediction"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """accurancy of prediction"""
    equality = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
