#!/usr/bin/env python3
"PCA, principal components analysis"

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset
    Arg:
       - X: numpy.ndarray of shape (n, d) where:
          - n: is the number of data points
          - d: is the number of dimensions in each point
       - ndim: new dimensionality of the transformed X

    Return:
       - T, a numpy.ndarray of shape (n, ndim) containing the
         transformed version of X
    """
    X_m = X - np.mean(X, axis=0)
    n = X.shape[0]
    # Compute covariance matrix
    # cov = np.cov(X.T)
    cov = np.dot(X_m.T, X_m) / (n-1)

    # eigen descomposition: eigenvalues, eigenvectors
    eigen_values, eigen_vectors = np.linalg.eig(cov)

    # eigen decomposition ordered in descending order
    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # dimensionality to be maintain
    eigen_values = eigen_values[:ndim]
    W = (-1) * eigen_vectors[:, :ndim]

    T = np.matmul(X_m, W)

    T = T.astype('float64')

    return T
