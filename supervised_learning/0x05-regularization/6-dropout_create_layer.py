#!/usr/bin/env python3
"""Dropout regularization of a model"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Forward propagation using Dropout
    prev: output prev layyer
    n: number of nodes
    activation: activation function
    keep_prob: probability that a node will be kept
    return: output of the new layer
    """
    dropout = tf.layers.Dropout(keep_prob)
    k_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=k_init,
                            kernel_regularizer=dropout)
    return layer(prev)
