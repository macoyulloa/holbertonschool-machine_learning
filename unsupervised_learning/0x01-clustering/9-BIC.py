#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """ find the best number of clusters for a GMM using the Bayesian
        Information Criterion:
    Arg:
        - X: is a numpy.ndarray of shape (n, d) containing the data set
        - kmin: the minimum number of clusters to check for (inclusive)
        - kmax: the maximum number of clusters to check for (inclusive)
        - iterations: maximum number of iterations for the EM algorithm
        - tol: the tolerance for the EM algorithm
        - verbose: is a boolean that determines if the EM algorithm
                    should print information to the standard output

    Returns: (best_k, best_result, l, b), or (None, None, None, None)
        - best_k: is the best value for k based on its BIC
        - best_result: is tuple containing pi, m, S
            - pi: np.shape (k,) has the cluster priors for the best num
                of clusters
            - m: np.shape (k, d) has the centroid means for the best
                number of clusters
            - S: np.hape (k, d, d) has the covariance matrices for the best
            number of clusters
        - l: np shape (kmax - kmin + 1) has the log likelihood for each
            cluster size tested
        - b: np shape (kmax - kmin + 1) containing the BIC value for
            each cluster size tested
            Use: BIC = p * ln(n) - 2 * l
                - p: number of parameters required for the model
                - n: num of data points used to create the model
                - l: is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None, None)
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return (None, None, None, None)
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return (None, None, None, None)
    if kmin >= kmax:
        return (None, None, None, None)
    if type(iterations) != int or iterations <= 0:
        return (None, None, None, None)
    if type(tol) != float or tol <= 0:
        return (None, None, None, None)
    if type(verbose) != bool:
        return None, None, None, None

    k_results, results, l_totals, b_totals = [], [], [], []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S, g, l = expectation_maximization(
            X, k, iterations, tol, verbose)
        k_results.append(k)
        results.append((pi, m, S))
        l_totals.append(l)
        bic = p * np.ln(n) - 2 * l
        b_totals.append(bic)
    b_totals = np.asarray(b_totals)
    best_b = np.argmin(b_totals)

    return (k_results[best_b], results[best_b], l_totals, b_totals)
