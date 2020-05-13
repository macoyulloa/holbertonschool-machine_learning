#!/usr/bin/env python3
"PCA, principal components analysis"

import numpy as np


def pca(X, var=0.96):
    """Performs PCA on a dataset
    Arg:
       - X: numpy.ndarray of shape (n, d) where:
          - n: is the number of data points
          - d: is the number of dimensions in each point
          - all dimensions have a mean of 0 across all data points
       - var: fraction of the variance that the PCA should be maintain

    Return:
       - weights: W, that maintains var fraction of X‘s original variance
    """
    n = X.shape[0]
    # Compute covariance matrix
    # cov = np.cov(X.T)
    cov = np.dot(X.T, X) / (n-1)

    # eigen descomposition: eigenvalues, eigenvectors
    eigen_values, eigen_vectors = np.linalg.eig(cov)

    # eigen decomposition ordered in descending order
    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    sum_eigen_vals = np.sum(eigen_values)
    # retention of the information per eigen value
    variance_ret = eigen_values / sum_eigen_vals
    # acumulation of the retention per iegen value
    acum_variance = np.cumsum(variance_ret)

    # fraction of the variance that the PCA should maintain
    r = 0
    for i in acum_variance:
        r += 1
        if i > var:
            break
    new_eigen_vecs = eigen_vectors[:, :r]

    return (-1) * new_eigen_vecs
