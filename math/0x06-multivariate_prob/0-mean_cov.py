#!/usr/bin/env python3
"""Estadistics: covariance and men"""

import numpy as np


def mean_cov(X):
    """ calculates the mena and covariance of a data set
    Arg:
       - X: numpy.ndarray of shape (n, d) containing the data set
            n: number of data points
            d: number of dimensions in each data point

    Returns: mean, cov
       - mean: numpy.ndarray (1, d) containing the mean of the data set
       - cov: numpy.ndarray (d, d) with covariance matrix of the data set
    """
    if (not type(X) == np.ndarray) or (len(X.shape) != 2):
        raise TypeError("X must be a 2D numpy.ndarray")
    n = X.shape[0]
    if n < 2:
        raise ValueError("X must contain multiple data points")

    d = X.shape[1]
    mean = np.mean(X, axis=0).reshape(1, d)

    X = X - mean
    cov = ((np.dot(X.T, X)) / (n - 1))

    return mean, cov
