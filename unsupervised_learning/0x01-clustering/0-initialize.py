#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means:
    Arg:
        - X: np.ndarray shape (n, d) dataset that will be used
             for K-means clustering
            - n number of data points
            - d number of dimensions for each data point
        - k is a positive integer containing the number of clusters
        - cluster centroids: initialized with a multivariate uniform
                            distribution along each dimension in d

    Returns: np.ndarray of shape (k, d)
        - Initialized centroids for each cluster, or None on failure
    """
    n, d = X.shape

    try:
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        return np.random.uniform(X_min, X_max, size=(k, d))

    except Exception:
        return None
