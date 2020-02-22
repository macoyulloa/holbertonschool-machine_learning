#!/usr/bin/env python3
"""Converts a numeric label vector into a one-hot matrix"""

import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label into a vector"""
    hot_encode = np.zeros((Y.size, Y.max() + 1))
    hot_encode[Y, np.arange(Y.size)] = 1
    return hot_encode
