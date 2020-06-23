#!/usr/bin/env python3
"""Generative Adversarial Networks"""

import tensorflow as tf


def generator(Z):
    """ simple generator network for MNIST digits
    All variables in the network should have the scope
    generator with reuse=tf.AUTO_REUSE

    Arg:
        - Z is a tf.tensor containing the input to the generator network

    Returns: X, a tf.tensor containing the generated image
    """
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        layer_1 = tf.layers.Dense(units=128, activation=tf.nn.relu,
                                  name="layer_1")(Z)

        X = tf.layers.Dense(units=784, activation=tf.nn.sigmoid,
                            name="layer_2")(layer_1)

    return X
