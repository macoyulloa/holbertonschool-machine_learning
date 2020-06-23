#!/usr/bin/env python3
"""Generative Adversarial Networks"""

import tensorflow as tf


def discriminator(X):
    """ discriminator network for MNIST digits:
    All variables in the network should have the scope
    dicriminator with reuse=tf.AUTO_REUSE

    Arg:
        - X is a tf.tensor containing the input to the discriminator net

    Returns:
        - Y, tf.tensor containing the classification made by the discriminator
    """
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        layer_1 = tf.layers.Dense(units=128, activation=tf.nn.relu,
                                  name="layer_1")(X)

        Y = tf.layers.Dense(units=1, activation=tf.nn.sigmoid,
                            name="layer_2")(layer_1)

    return Y
