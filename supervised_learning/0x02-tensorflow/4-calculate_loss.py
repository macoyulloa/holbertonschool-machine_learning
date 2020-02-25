#!/usr/bin/env python3
"""loss function"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """loss function"""
    loss = tf.losses.softmax_cross_entropy(
        y, y_pred, reduction=tf.losses.Reduction.MEAN)
    return loss
