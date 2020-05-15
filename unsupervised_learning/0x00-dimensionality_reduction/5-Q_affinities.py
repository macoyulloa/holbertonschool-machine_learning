#!/usr/bin/env python3
"""t-SNE method"""

import numpy as np


def Q_affinities(Y):
    """calculates Q affinities:
    Arg:
      - Y: np.ndarray (n, ndim) with the low dimensional transformation of X
         n is the number of data points
         ndim  is the new dimensional representation of X

    Returns: (Q, num)
      - Q: numpy.ndarray of shape (n, n) containing the Q affinities
      - num: numpy.ndarray (n, n) containing the numerator of the Q affinities
    """
    n, ndim = Y.shape

    sum_Y = np.sum(np.square(Y), axis=1)
    distances = (np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))

    num = 1. / (1. + distances)
    np.fill_diagonal(num, 0.)
    Q = num / np.sum(num)

    return (Q, num)
