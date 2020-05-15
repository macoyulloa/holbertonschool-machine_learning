#!/usr/bin/env python3
"""t-SNE method"""

import numpy as np


def cost(P, Q):
    """calculates the cost if the t-SNE
    Arg:
      - P: numpy.ndarray of shape (n, n) containing the P affinities of X
      - Q: is a numpy.ndarray of shape (n, n) containing the Q affinities

    Returns:
      - C, the cost of the transformation
    """
    C = np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12)))
    return (C)
