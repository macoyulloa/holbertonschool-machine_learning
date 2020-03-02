#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """training operation for a neural network in tensorflow"""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return (optimizer)
