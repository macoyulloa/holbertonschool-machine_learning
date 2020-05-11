#!/usr/bin/env python3
"""multinormal of a matrix"""

import numpy as np


class MultiNormal:
    "represents a Multivariate Normal distribution"

    def __init__(self, data):
        """initialized the variables of the class
        Arg:
        - data: numpy.ndarray of shape (d, n) containing the data set
              - n is the number of data points
              - d is the number of dimensions in each data point

        Public instance variables:
          - mean: numpy.ndarray of shape (d, 1) containing the mean of data
          - cov: numpy.ndarray (d, d) containing the covariance matrix data
        """
        if (not type(data) == np.ndarray) or (len(data.shape) != 2):
            raise TypeError("data must be a 2D numpy.ndarray")
        n = data.shape[0]
        if n < 2:
            raise ValueError("data must contain multiple data points")

        d = data.shape[1]
        self.mean = np.mean(data, axis=0).reshape(d, 1)

        X = data - mean
        self.cov = ((np.dot(X.T, X)) / (n - 1))
