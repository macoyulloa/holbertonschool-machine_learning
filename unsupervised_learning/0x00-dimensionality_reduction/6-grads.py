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

    pq_diff = P - Q  # NxN matrix
    pq_expanded = np.expand_dims(pq_diff, 2)  #NxNx1
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  #NxNx2
    dY = 4. * (pq_expanded * y_diffs).sum(1)  #Nx2

    return (dY, Q)
