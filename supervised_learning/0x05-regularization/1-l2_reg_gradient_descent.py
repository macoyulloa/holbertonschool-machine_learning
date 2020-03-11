#!/usr/bin/env python3
"""regularization of a model"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weight and biases using gradient des L2 reg
    Y: is a one-hot (classes, m)
    lambtha: regularization param
    weights: dic with w and b
    cache: dic of outputs of each layer
    alpha: learning rate
    L: num of layers
    """
    weights1 = weights.copy()
    m = Y.shape[1]
    A_L = cache['A'+str(L)]
    A_prev = cache['A'+str(L - 1)]
    W_L = weights1['W'+str(L)]
    b_L = weights1['b'+str(L)]
    dz_list = {}
    dz_L = A_L - Y
    dz_list['dz'+str(L)] = dz_L
    dw_L = ((1/m) * np.matmul(A_prev, dz_L.T)).T + (lambtha/m * W_L)
    db_L = (1/m) * np.sum(dz_L, axis=1, keepdims=True)
    weights['W'+str(L)] = W_L - (alpha * dw_L)
    weights['b'+str(L)] = b_L - (alpha * db_L)

    for l in range(L - 1, 0):
        A_curr = cache['A'+str(l)]
        A_bef = cache['A'+str(l-1)]
        W_curr = weights1['W'+str(l)]
        W_next = weights1['W'+str(l+1)]
        b_curr = weights1['b'+str(l)]
        dz1 = np.matmul(W_next.T, dz_list['dz'+str(l+1)])
        dz2 = (1 - A_curr**2)
        dz = dz1 * dz2
        dw = (1/m) * np.matmul(dz, A_prev.T) + (lambtha/m * W_L)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        dz_list['dz'+str(l)] = dz
        weights['W'+str(l)] = W_curr - (alpha * dw)
        weights['b'+str(l)] = b_curr - (alpha * db)
