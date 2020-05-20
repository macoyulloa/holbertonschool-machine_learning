#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model:
    Arg:
        - X is a numpy.ndarray of shape (n, d) containing the data set
        - k is a positive integer containing the number of clusters

    Returns: (pi, m, S), or (None, None, None) on failure
        - pi: np.ndarray shape (k,) with the priors for each cluster,
                initialized evenly
        - m: np.ndarray shape (k, d) with the centroid means for
                each cluster, initialized with K-means
        - S: np.ndarray shape (k, d, d) containing the covariance
                matrices for each cluster, identity matrices
    """
    try:
        _, d = X.shape
        pi = np.repeat(1/k, k)
        m, _ = kmeans(X, k)
        S = np.tile(np.identity(d), k)
        S= np.reshape(S, (k, d, d))
        return (pi, m, S)

    except Exception:
        return (None, None, None)
