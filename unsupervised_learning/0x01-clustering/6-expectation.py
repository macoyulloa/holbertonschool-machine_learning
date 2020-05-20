#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ calculates the expectation step in the EM algorithm for a GMM:
    Arg:
        - X: np.ndarray of shape (n, d) containing the data set
        - pi: np.ndarray of shape (k,) with the priors per cluster
        - m: np.ndarray of shape (k, d) with centroid means per cluster
        - S: np.ndarray shape (k, d, d) with cov matrices per cluster

    Returns: (g, l), or (None, None) on failure
        - g: np.ndarray of shape (k, n) with the posterior
            probabilities for each data point in each cluster
        - l: is the total log likelihood
    """
    n, _ = X.shape
    k = m.shape[0]
    gauss_p = np.zeros((k, n))

    for i in range(k):
        mi = m[i, :]
        Si = S[i, :, :]
        P = pdf(X, mi, Si)
        print(P)
        gauss_p[i, :] = P * pi[i]

    g = gauss_p / np.sum(gauss_p)
    l = np.sum(np.log(gauss_p))

    return (g, l)
