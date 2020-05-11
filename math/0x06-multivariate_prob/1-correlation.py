#!/usr/bin/env python3
"""Estadistics: covariance,  mean and correlation"""

import numpy as np


def correlation(C):
    """ calculates the correlation of a matrix
    Arg:
       - X: numpy.ndarray of shape (n, n) containing the covariance matrix
            n: number of data points

    Returns:
       - numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if (not type(C) == np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if (len(C.shape) != 2):
        raise ValueError("C must be a 2D square matrix")
    d1, d2 = C.shape[0], C.shape[1]
    if d1 != d2:
        raise ValueError("C must be a 2D square matrix")

    correlation = np.corrcoef(C)

    return correlation
