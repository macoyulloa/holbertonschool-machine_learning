#!/usr/bin/env python3
"""Optimization tasks"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch normalization using tensorflow"""
    layer = tf.layers.Dense(units=n, activation=activation)
    m, var = tf.nn.moments(layer(prev), axes=[0])
    norm = tf.nn.batch_normalization(layer(prev), mean=m, variance=var, offset=None, scale=None, variance_epsilon=1e-8)
    return norm
