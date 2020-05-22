#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np


def kmeans(X, k, iterations=1000):
    """ performs K-means on a dataset
    Arg:
        - X: np.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - k: positive int containing the number of clusters
        - iterations: positive int with the max number of iterations
                        that should be performed

    Returns: (C, clss) or (None, None) on failure
        - C: np.ndarray (k, d) with the centroid means for each cluster
        - clss: np.ndarray (n,) with the index of the cluster in C that
                each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    n, d = X.shape

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    # initialized the centroids points
    C = np.random.uniform(X_min, X_max, size=(k, d))
    # clustering my data
    for i in range(iterations):
        C_copy = np.copy(C)
        # distances of the x's between each centroid, k
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        # matrix with the centroid per xi
        clss = np.argmin(distances, axis=0)

        # moving my centroid depends on the average mean based on the
        # group of x's per cluster
        for k in range(C.shape[0]):
            # if not found any x of the centroid need to be reinitialized
            if (X[clss == k].size == 0):
                C[k, :] = np.random.uniform(X_min, X_max, size=(1, d))
                # look after the group of x's of each cluster per centroid
            else:
                C[k, :] = (X[clss == k].mean(axis=0))
                # if there is not change in the centroids breaks the cycle

        if (C_copy == C).all():
            return (C, clss)

    return (C, clss)
