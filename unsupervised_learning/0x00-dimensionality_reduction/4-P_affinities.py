#!/usr/bin/env python3
"""t-SNE method"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """calculates the symmetric P affinities of a data set:
    Arg:
      - X: np.ndarray (n, d) containing the dataset to be transformed by t-SNE
         n is the number of data points
         d is the number of dimensions in each point
      - perplexity: that all Gaussian distributions should have
      - tol: maximum tolerance allowed (inclusive) for the difference in
             Shannon entropy from perplexity for all Gaussian distri

    Returns: (P)
      - P: np.ndarray (n, n) containing the symmetric P affinities
    """
    n, d = X.shape

    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        Hi, P[i, 1:] = HP(D[i, 1:], betas[i])

    return (P)
