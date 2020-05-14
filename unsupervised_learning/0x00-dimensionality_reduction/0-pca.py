#!/usr/bin/env python3
"PCA, principal components analysis"

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset
    Arg:
       - X: numpy.ndarray of shape (n, d) where:
          - n: is the number of data points
          - d: is the number of dimensions in each point
          - all dimensions have a mean of 0 across all data points
       - var: fraction of the variance that the PCA should be maintain

    Return:
       - weights: W, that maintains var fraction of X‘s original variance
    """
    u, s, vh = np.linalg.svd(X)
    total_variance = np.cumsum(s) / np.sum(s)
    r = (np.argwhere(total_variance >= var))[0, 0]
    print(r)
    w = vh[:r + 1].T

    return w
