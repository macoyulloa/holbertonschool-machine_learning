#!/usr/bin/env python3
"""t-SNE method"""

import numpy as np


def P_init(X, perplexity):
    """initializes variables used to calculate the P affinities in t-SNE:
    Arg:
      - X: np.ndarray (n, d) containing the dataset to be transformed by t-SNE
         n is the number of data points
         d is the number of dimensions in each point
      - perplexity: that all Gaussian distributions should have

    Returns: (D, P, betas, H)
      - D: np.ndarray (n, n) calculates pairwise distance between data points
      - P: np.ndarray (n, n) initialized 0‘s that will contain the P affinities
      - betas: np.ndarray (n, 1) initialized to all 1’s that contain beta values
      - H is the Shannon entropy for perplexity perplexity
    """
    n, d = X.shape

    sum_X = np.sum(np.square(X), axis=1)
    D = (np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X))

    P = np.zeros([n, n], dtype = 'float64')

    betas = np.ones([n, 1], dtype = 'float64')

    H = np.log2(perplexity)

    return (D, P, betas, H)
