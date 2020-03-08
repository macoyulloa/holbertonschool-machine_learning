#!/usr/bin/env python3
"""Optimization tasks"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch normalization using tensorflow"""
    kernel_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None,
                            kernel_initializer=kernel_init)
    m, var = tf.nn.moments(layer(prev), axes=[0])
    beta = tf.get_variable(name="beta", shape=[n],
                           initializer=tf.zeros_initializer(),
                           trainable=True)
    gamma = tf.get_variable(name="gamma", shape=[n],
                            initializer=tf.ones_initializer(),
                            trainable=True)
    norm = tf.nn.batch_normalization(layer(prev), mean=m,
                                     variance=var, offset=beta,
                                     scale=gamma, variance_epsilon=1e-8)
    return activation(norm)
