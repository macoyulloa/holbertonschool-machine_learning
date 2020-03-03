#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learming rate decay"""
    alpha = (1 / (1 + (decay_rate * decay_step)))
    return (alpha)
