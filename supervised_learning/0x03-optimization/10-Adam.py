#!/usr/bin/env python3
"""Optimization tasks"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Adam optimization algorithm"""
    optim = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).miniize(loss)
    return (optim)
