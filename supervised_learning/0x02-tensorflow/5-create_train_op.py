#!/usr/bin/env python3
"""training operation network"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """training ope for the network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return optimizer
