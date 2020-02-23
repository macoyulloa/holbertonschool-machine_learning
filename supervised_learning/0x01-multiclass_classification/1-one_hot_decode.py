#!/usr/bin/env python3
"""Converts one-hot matrix into a numerical vector"""

import numpy as np


def one_hot_decode(one_hot):
    """converts one-hot matrix inot a numerical vector"""
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    vector = np.arange(one_hot.shape[1])
    for i in range(len(one_hot)):
        for j in range(len(one_hot[0])):
            if one_hot[i, j] == 1:
                vector[j] = i
    if one_hot.shape[0] < vector.max() + 1:
        return None
    return vector
