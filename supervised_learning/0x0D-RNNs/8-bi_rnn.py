#!/usr/bin/env python3
"""Biderectional Recurrent Neural Network"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """performs forward propagation for a deep RNN:

    Arg:
        - bi_cellinstance of BidirectinalCell that will be
            used for the forward propagation
        - X: is the data, np.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        - h_0: initial hidden state, np.ndarray of shape (m, h)
            - h is the dimensionality of the hidden state
        - h_t: is the initial hidden state in the backward direction,
            given as a numpy.ndarray of shape (m, h)

        Returns: H, Y
        - H: np.ndarray containing all of the concat hidden states
        - Y: np.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    H_forward = np.zeros((t+1, m, h))
    H_backward = np.zeros((t+1, m, h))
    H_forward[0] = h_0
    H_backward[t] = h_t
    for t_i in range(t):
        H_forward[t_i+1] = bi_cell.forward(H_forward[t_i], X[t_i])
    for t_j in range(t-1, -1, -1):
        H_backward[t_j-1] = bi_cell.backward(H_backward[t_j], X[t_j])
    H = np.concatenate((H_forward[1:], H_backward[0:t]), axis=2)

    Y = bi_cell.output(H)

    return H, Y
