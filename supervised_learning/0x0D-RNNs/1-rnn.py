#!/usr/bin/env python3
"""Recurrent Neural Network"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """ performs forward propagation for a simple RNN:

        Arg:
        - rnn_cell: instande of RNNCell used for the forward propag
        - X: is the data, np.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        - h_0: initial hidden state, np.ndarray of shape (m, h)
            - h is the dimensionality of the hidden state

        Returns: H, Y
        - H: np.ndarray containing all of the hidden states
        - Y: np.ndarray containing all of the outputs
    """
    time_steps, m, i = X.shape
    _, h = h_0.shape

    Y = []
    H = np.zeros((time_steps+1, m, h))
    H[0, :, :] = h_0
    for t_step in range(time_steps):
        h, y = rnn_cell.forward(H[t_step], X[t_step])
        H[t_step+1, :, :] = h
        Y.append(y)
    Y = np.asarray(Y)
    return H, Y
