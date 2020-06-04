#!/usr/bin/env python3
"class BayesianOptimization"

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """Bayesian optimization noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ Initialized the variables of the Bayesian optimization

        Arg:
            - f is the black-box function to be optimized
            - X_init is a numpy.ndarray of shape (t, 1) representing
                the inputs already sampled with the black-box function
            - Y_init is a numpy.ndarray of shape (t, 1) representing
                outputs of the black-box function for each input in X_init
                - t is the number of initial samples
            - bounds is a tuple of (min, max) representing the bounds of
                the space in which to look for the optimal point
            - ac_samples is the number of samples that should be analyzed
                during acquisition
            - l is the length parameter for the kernel
            - sigma_f is the standard deviation given to the output of the
                black-box function
            - xsi is the exploration-exploitation factor for acquisition
            - minimize is a bool determining whether optimization should be
                performed for minimization (True) or maximization (False)

        Public instance attributes:
            - f: the black-box function
            - gp: an instance of the class GaussianProcess
            - X_s: a numpy.ndarray of shape (ac_samples, 1) containing all
                acquisition sample points, evenly spaced between min and max
            - xsi: the exploration-exploitation factor
            - minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.random.uniform(bounds[0], bounds[1], (ac_samples, 1))
        self.xsi = xsi
        self.minimize = minimize
        self.X = X_init

    def acquisition(self):
        """ calculates the next best sample location.
        Uses the Expected Improvement acquisition function

        Returns: X_next, EI
            - X_next: is a numpy.ndarray of shape (1,) representing the
                    next best sample point
            - EI: is a numpy.ndarray of shape (ac_samples,) containing
                    the expected improvement of each potential sample
        """
        mu_sample, sigma_sample = self.gp.predict(self.X_s)
        # sigma = sigma.reshape(-1, 1)
        # print(sigma.shape)
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        mu_sample_opt = np.max(mu_sample)
        # mu_x, sigma_x = self.gp.predict(self.X_s[np.argmax(mu_sample_opt)])

        with np.errstate(divide='warn'):
            imp = mu_sample - mu_sample_opt - self.xsi
            Z = imp / sigma_sample
            part1 = (imp * norm.cdf(Z))
            part2 = (sigma_sample * norm.pdf(Z))
            ei = part1 + part2
            ei[sigma_sample == 0.0] = 0.0
        X_next = np.random.uniform(-np.pi, 2*np.pi, 1)
        return X_next, ei

    def optimize(self, iterations=100):
        """ that optimizes the black-box function:

        Arg:
        - iterations: is the maximum number of iterations to perform

        If the next proposed point is one that has already been sampled,
        optimization should be stopped early

        Returns: X_opt, Y_opt
            - X_opt: is a numpy.ndarray of shape (1,) representing the
                    optimal point
            - Y_opt: is a numpy.ndarray of shape (1,) representing the
                    optimal function value
        """
        Y_s = self.f(self.X_s)

        for i in range(iterations):
            self.gp.fit(self.X_s, Y_s)
            X_opt, EI = self.acquisition()
            Y_opt = self.f(X_opt)
            if np.any(self.X_s == X_opt):
                return (X_opt, Y_opt)
            self.gp.update(X_opt, Y_opt)

        return (X_opt, Y_opt)
