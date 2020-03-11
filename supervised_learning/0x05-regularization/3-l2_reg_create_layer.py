#!/usr/bin/env python3
"""regularization of a model"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a layer includes L2 reg
        prev: is a tensor with the output of the prev layer
        n: is the number of nodes
        activation: function that should be used on the layer
        lambtha: is the L2 regularization parameter
        return: output of a new layer
    """
    l2_reg = tf.contrib.layers.l2_regularizer(lambtha)
    k_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=k_init,
                            kernel_regularizer=l2_reg)
    return layer(prev)
