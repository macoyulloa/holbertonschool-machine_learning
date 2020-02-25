#!/usr/bin/env python3
"""create a layers"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """crate the layers of the model and variables"""
    tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, activation, name='layer')
    return layer
