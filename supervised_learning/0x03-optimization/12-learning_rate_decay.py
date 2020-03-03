#!/usr/bin/env python3
"""Optimization tasks"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learming rate decay"""
    learning_rate = tf.train.inverse_time_decay(alpha, global_step,
                                                decay_step, decay_rate)
    return (learning_rate)
