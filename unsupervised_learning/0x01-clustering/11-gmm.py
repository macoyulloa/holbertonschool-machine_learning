#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import sklearn.mixture


def gmm(X, k):
    """ that calculates a GMM from a dataset:

    Arg:
        - X is a numpy.ndarray of shape (n, d) containing the dataset
        - k is the number of clusters

    Returns: pi, m, S, clss, bic
        - pi: np.ndarray of shape (k,) containing the cluster priors
        - m: np.ndarray of shape (k, d) containing the centroid means
        - S: np.ndarray of shape (k, d, d) with the covariance matrices
        - clss: np.ndarray of shape (n,) containing the cluster indices
                for each data point
        - bic: np.ndarray of shape (kmax - kmin + 1) containing the
                BIC value for each cluster size tested
    """
    g_mixture = sklearn.mixture.GaussianMixture(n_components=k)
    g = g_mixture.fit(X)
    m = g.means_
    S = g.covariances_
    pi = g.weights_
    clss = g_mixture.predict(X)
    bic = g_mixture.bic(X)

    return (pi, m, S, clss, bic)
