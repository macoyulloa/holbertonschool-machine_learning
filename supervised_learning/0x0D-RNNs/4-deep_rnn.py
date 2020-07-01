#!/usr/bin/env python3
"""Recurrent Neural Network"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ performs forward propagation for a deep RNN:

        Arg:
        - rnn_cells is a list of RNNCell instances of length l
            that will be used for the forward propagation
            - l is the number of layers
        - X: is the data, np.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        - h_0: initial hidden state, np.ndarray of shape (l, m, h)
            - h is the dimensionality of the hidden state

        Returns: H, Y
        - H: np.ndarray containing all of the hidden states
        - Y: np.ndarray containing all of the outputs
    """
    time_steps, m, i = X.shape
    Y = []

    H = h_0
    for t_step in range(time_steps):
        for layer in range(len(rnn_cells)):
            if t_step == 0:
                h, y = rnn_cells[layer].forward(h_0[layer], X[0])
            else:
                h, y = rnn_cells[layer].forward(h[layer], X[t_step])
        H = np.concatenate((H, h))
        Y.append(y)

    Y = np.asarray(Y)
    return H, Y
