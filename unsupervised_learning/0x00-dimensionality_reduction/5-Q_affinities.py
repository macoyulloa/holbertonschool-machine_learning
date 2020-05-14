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

    exp_distances = np.exp(-distances)
    np.fill_diagonal(exp_distances, 0.)
    Q = exp_distances / np.sum(exp_distances)

    return (Q, exp_distances)
