#!/usr/bin/env python3
"""t-SNE method"""

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """calculates t-SNE transformation:
    Arg:
       - X: np.ndarray shape (n, d) containing the dataset to be transform:
             - n is the number of data points
             - d is the number of dimensions in each point
       - ndims is the new dimensional representation of X
       - idims is the intermediate dimensional representation of X after PCA
       - perplexity is the perplexity
       - iterations is the number of iterations
       - lr is the learning rate

    Returns:
       - Y: np.ndarray of shape (n, ndim) containing the optimized low
            dimensional transformation of X
    """
    n, d = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8

    X = pca(X, idims)
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))

    P = P_affinities(X, perplexity)
    # early exageration
    P = P * 4

    for i in range(iterations):

        dY, Q = grads(Y, P)
        if i < (20):
            momentum = initial_momentum
        else:
            momentum = final_momentum

        # perform the update
        iY = momentum * iY + lr * dY
        Y = Y - iY

        # print cost of the T SEN model
        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format((i+1), C))

        # perform the aerly exagerations for first 100 iterations
        if (i + 1) == 100:
            P = P / 4

    return (Y)
