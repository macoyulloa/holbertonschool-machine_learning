#!/usr/bin/env python3
""" L2 Regularization """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network
    Y: is a one-hot (classes, m)
    lambtha: regularization param
    weights: dic with w and b
    cache: dic of outputs of each layer
    alpha: learning rate
    L: num of layers
    """

    w_copy = weights.copy()
    m = Y.shape[1]

    for ln in reversed(range(L)):
        if ln == L-1:
            dz = cache["A"+str(ln+1)] - Y
            dw = (np.matmul(cache["A"+str(ln)], dz.T) / m).T
            dw_regu = dw + (lambtha/m) * w_copy["W"+str(ln+1)]
            db = np.sum(dz, axis=1, keepdims=True) / m
        else:
            dz1 = np.matmul(w_copy["W"+str(ln+2)].T, dz_curr)
            dz2 = 1-cache["A"+str(ln+1)]**2
            dz = dz1 * dz2
            dw = np.matmul(dz, cache["A"+str(ln)].T) / m
            dw_regu = dw + (lambtha/m) * w_copy["W"+str(ln+1)]
            db = np.sum(dz, axis=1, keepdims=True) / m
        weights["W"+str(ln+1)] = (w_copy["W"+str(ln+1)] - (alpha * dw_regu))
        weights["b"+str(ln+1)] = w_copy["b"+str(ln+1)] - alpha * db
        dz_curr = dz
