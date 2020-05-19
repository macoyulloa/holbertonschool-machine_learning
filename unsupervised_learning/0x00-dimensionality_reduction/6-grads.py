#!/usr/bin/env python3
"""t-SNE method"""

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """calculates gradients of Y
    Arg:
      - Y: np.ndarray (n, ndim) with the low dimensional transformation of X
         n is the number of data points
         ndim  is the new dimensional representation of X

      - P: numpy.ndarray of shape (n, n) containing the P affinities of X

    Returns: (dY, Q)
      - dY: numpy.ndarray of shape (n, n) containing the gradients of Y
      - Q: numpy.ndarray of shape (n, n) containing the Q affinities
    """
    n, ndim = Y.shape

    Q, num = Q_affinities(Y)
    pq_diff = P - Q
    # pq_diff_num = np.expand_dims((pq_diff * num).T, axis=-1)
    dY = np.zeros([n, ndim])
    for i in range(n):
        Y_diff = Y[i, :] - Y
        # dY[i, :] = np.sum((pq_diff_num[i, :] * Y_diff), axis=0)
        part1 = np.expand_dims((pq_diff[:, i] * num[:, i]).T, axis=-1)
        part2 = (Y[i, :] - Y)
        dY[i, :] = np.sum(part1 * part2, axis=0)

    return (dY, Q)
