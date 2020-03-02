#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices"""
    X_shuffle = np.random.permutation(X)
    Y_shuffle = np.random.permutation(Y)
    return (X_shuffle, Y_shuffle)
