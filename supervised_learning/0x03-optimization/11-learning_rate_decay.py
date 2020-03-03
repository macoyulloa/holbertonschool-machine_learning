#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learming rate decay"""
    alpha = (alpha / (1 + (decay_rate * int(global_step/decay_step))))
    return alpha
