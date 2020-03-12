#!/usr/bin/env python3
""" Dropout Regularization """

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ updates the weights and biases of a neural network
    X: array(nx, m)
    weights: dic with w and b
    keep_prob
    L: num of layers
    """
    cache = {}
    cache['A0'] = X
    for l in range(1, L + 1):
        W_curr = weights['W'+str(l)]
        b_curr = weights['b'+str(l)]
        A_prev = cache['A'+str(l-1)]
        z = (np.matmul(W_curr, A_prev)) + b_curr
        drop = np.random.binomial(1, keep_prob, size=z.shape)
        if l is L:
            t = np.exp(z)
            cache['A'+str(l)] = t/np.sum(t, axis=0, keepdims=True)
        else:
            cache['A' + str(l)] = np.tanh(z)
            cache['D' + str(l)] = drop
            cache['A' + str(l)] *= drop
            cache['A' + str(l)] /= keep_prob
    return (cache)
