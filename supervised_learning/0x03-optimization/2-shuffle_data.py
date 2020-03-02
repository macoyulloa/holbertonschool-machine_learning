#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def shuffle_data(X, Y):
    """permutates the data points in two matrices"""
    shuffle = np.random.permutation(len(X))
    return (X[shuffle], Y[shuffle])
