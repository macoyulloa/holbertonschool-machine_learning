#!/usr/bin/env python3
""" Gradient Descent with L2 Regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network
    using gradient descent with L2 regularization
    """
    w_copy = weights.copy()
    m = Y.shape[1]

    for l in reversed(range(L)):
        if l == L-1:
            dz = cache["A"+str(l+1)] - Y
            dw = (np.matmul(cache["A"+str(l)], dz.T) / m).T
        else:
            d1 = np.matmul(w_copy["W"+str(l+2)].T, dzp)
            d2 = 1-cache["A"+str(l+1)]**2
            dz = d1 * d2
            dw = np.matmul(dz, cache["A"+str(l)].T) / m
        dw_reg = dw + (lambtha/m) * w_copy["W"+str(l+1)]
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights["W"+str(l+1)] = (w_copy["W"+str(l+1)] - (alpha * dw_reg))
        weights["b"+str(l+1)] = w_copy["b"+str(l+1)] - alpha * db
        dzp = dz
