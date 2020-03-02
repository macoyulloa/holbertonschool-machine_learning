#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """gradient descent RMSProp optimization algorithm"""
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon).minimize(loss)
    return (optimizer)
