#!/usr/bin/env python3
"""Optimization tasks"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch normalization using tensorflow"""
    layer = tf.layers.Dense(units=n, activation=activation)
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    norm = tf.nn.batch_normalization(layer(prev), mean=mean,
                                     variance=variance, offset=None,
                                     scale=None, variance_epsilon=1e-3)
    return norm
