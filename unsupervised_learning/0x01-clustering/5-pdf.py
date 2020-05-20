#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np


def pdf(X, m, S):
    """ calculates the probability density function of a Gaussian distri
    Arg:
        - X: np.ndarray of shape (n, d) containing the data points
                whose PDF should be evaluated
        - m: np.ndarray of shape (d,) with the mean of the distribution
        - S: np.ndarray of shape (d, d) with the covariance of the distri

    Returns: (P), or (None) on failure
        - P: np.ndarray shape (n,) with the PDF values for each data point
                All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1] or X.shape[1] != S.shape[1]:
        return None

    n, d = X.shape
    X_m = X - m
    # covariance matrix inverted
    S_inv = np.linalg.inv(S)

    part1 = 1. / (np.sqrt(((2 * np.pi)**d * np.linalg.det(S))))
    # This einsum call calculates (x-m)T * S * (x-m) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', X_m, S_inv, X_m)
    part2 = np.exp(-fac / 2)
    P = part1 * part2
    P = np.maximum(P, 1e-300)
    return (P)
