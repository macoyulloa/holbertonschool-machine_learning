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

    # iterate each poiint per each k, cluster
    for i in range(k):
        P = pdf(X, m[i], S[i])
        gauss_p[i] = P * pi[i]
        # g_sum += gauss_p
    g = gauss_p / np.sum(gauss_p, axis=0)
    log_likelihood = np.sum(np.log(np.sum(gauss_p, axis=0)))

    return (g, log_likelihood)
