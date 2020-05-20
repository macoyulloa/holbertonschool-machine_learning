#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ optimum number of clusters by variance
    Arg:
        X: np.ndarray of shape (n, d) containing the data set
        kmin: positive int, minimum number of clusters to check
                for (inclusive)
        kmax: positive int, maximum number of clusters to check
        iterations: is a positive integer containing the maximum
                    number of iterations for K-means

    Returns: (results, d_vars) or (None, None) on failure
        - results: list, outputs of K-means for each cluster size
        - d_vars: list, with the difference in varianca from smallest
                    cluster size for each cluster size
    """
    try:
        results = []
        d_vars = []
        C_kmin, _ = kmeans(X, kmin)
        kmin_var = variance(X, C_kmin)
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k)
            results.append((C, clss))
            d_vars.append(kmin_var - variance(X, C))

        return (results, d_vars)

    except Exception:
        return (None, None)
