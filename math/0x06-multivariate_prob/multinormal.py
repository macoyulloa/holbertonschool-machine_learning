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
        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")

        d = data.shape[0]
        self.mean = (np.mean(data, axis=1)).reshape(d, 1)

        X = data - self.mean
        self.cov = ((np.dot(X, X.T)) / (n - 1))

    def pdf(self, x):
        """ calculates the PDF at a data point
        Arg:
           - x: np.ndarray of shape (d, 1) containing the data point
                whose PDF should be calculated
                - d: the number of dimensions of the Multinomial instance

        Returns: the value of the PDF
        """
        if not type(x) == np.ndarray:
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if (len(x.shape) != 2):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        if (x.shape[1] != 1) or (x.shape[0] != d):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        x_m = x - self.mean

        pdf1 = 1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(self.cov)))
        pdf2 = (np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2))

        pdf = pdf1 * pdf2

        return float(pdf)
