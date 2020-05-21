#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                            verbose=False):
    """ performs the expectation maximization for a GMM:
    Arg:
        - X: np.ndarray shape (n, d) containing the data set
        - k: is a positive integer containing the number of clusters
        - iterations: is a positive integer containing the maximum
                        number of iterations for the algorithm
        - tol: non-negative float, tolerance of the log likelihood,
                used to determine early stopping i.e. if the difference
                is less than or equal to tol you should stop it
        - verbose: boolean that determines if you should print inf
                about the algorithm

    Returns: (pi, m, S, g, l), or (None, None, None, None, None)
        - pi: np.ndarray of shape (k,) containing the priors for
                each cluster
        - m: np.ndarray of shape (k, d) containing the centroid
                means for each cluster
        - S: np.ndarray of shape (k, d, d) containing the covariance
                matrices for each cluster
        - g: np.ndarray of shape (k, n) containing the probabilities
                for each data point in each cluster
        - l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None, None, None)
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return (None, None, None, None, None)
    if type(iterations) != int or iterations <= 0:
        return (None, None, None, None, None)
    if type(tol) != float or tol <= 0:
        return (None, None, None, None, None)

    for i in range(iterations):
        #if abs(l - l_past) <= tol:
        #    print("Log Likelihood after {} iterations: {}".format(
        #        i, l))
        #    return (pi, m, S, g, l)
        pi, m, S = initialize(X, k)
        g, l = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if (verbose == True):
            if (i % 10 == 0) or (i == 0):
                print("Log Likelihood after {} iterations: {}".format(
                    i, l))
        # l_past = l
    return (pi, m, S, g, l)
