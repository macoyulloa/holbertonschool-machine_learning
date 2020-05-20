#!/usr/bin/env python3

import numpy as np


def variance(X, C):
    """ calculates the total intra-cluster variance for a data set
    Arg:
        - X: np.ndarray shape (n, d) containing the data set
        - C: np.ndarray shape(k, d) with the centroid means for each cluster

    Returns: var, or None on failure
        - var: is the total variance
    """
    n, d = X.shape
    try:
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        # selectin the minimun distances depends on the number of
        # clusters, K
        min_distances = np.min(distances, axis=0)
        var = np.sum(min_distances ** 2)
        return var

    except Exception:
        return None
