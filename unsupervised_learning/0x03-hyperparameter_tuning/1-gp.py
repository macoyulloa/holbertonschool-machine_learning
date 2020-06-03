#!/usr/bin/env python3
"Class GaussianProces: represents a noiseless 1D Gaussian process"

import numpy as np


class GaussianProcess():
    "represents a noiseless 1D Gaussian process"

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Initialized the variables

        Arg:
        - X_init: np.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
        - Y_init: np.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
        - t: is the number of initial samples
        - l: is the length parameter for the kernel
        - sigma_f: is the standard deviation given to the output of the
                    black-box function
        Public instance attributes:
        - X, Y, l, and sigma_f : with the respective constructor inputs
        - K, representing the current covariance kernel matrix for the
            Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ calculates the covariance kernel matrix between two matrices

        Arg:
            - X1 is a numpy.ndarray of shape (m, 1)
            - X2 is a numpy.ndarray of shape (n, 1)

        Returns: covariance kernel matrix as a np.ndarray of shape (m, n)
        """
        sqdist = (np.sum(X1**2, 1).reshape(-1, 1)) + \
            (np.sum(X2**2, 1)) - (2 * np.dot(X1, X2.T))
        return (self.sigma_f**2) * (np.exp(-1/(2 * (self.l**2)) * sqdist))

    def predict(self, X_s):
        """ predicts the mean and standard desv of a points in GP

        Arg:
        - X_s: is a numpy.ndarray of shape (s, 1) with all of the points
            whose mean and standard deviation should be calculated
            - s: is the number of sample points

        Returns: mu, sigma
            - mu: np.ndarray of shape (s,) containing the mean for
                    each point in X_s, respectively
            - sigma: np.ndarray of shape (s,) with the standard
                    desviation for each point in X_s, respectively
        """
        # mu = L.dot(np.random.randn(*Y.shape))
        # var = np.abs(L.dot(np.random.randn(*Y.shape))) + 0.01
        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = np.reshape(mu_s, -1)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma_s = np.diagonal(cov_s)

        return mu_s, sigma_s
