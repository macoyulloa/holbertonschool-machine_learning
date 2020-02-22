#!/usr/bin/env python3
"""Converts a numeric label vector into a one-hot matrix"""

import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label into a vector"""
    if len(Y) == 0 or len(Y) != classes:
        return None
    hot_encode = np.zeros((classes, Y.shape[0]))
    hot_encode[Y, np.arange(Y.size)] = 1
    return hot_encode
