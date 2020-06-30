#!/usr/bin/env python3
"""Recurrent Neural Network"""

import numpy as np


class RNNCell:
    """RNNCell class, vanila model"""

    def __init__(self, i, h, o):
        """ initialized the variables

        Arg:
        - i: is the dimensionality of the data
        - h: is the dimensionality of the hidden state
        - o: is the dimensionality of the outputs

        - Public instance attributes Wh, Wy, bh, by that represent
            the weights and biases of the cell
            - Wh and bh: for the concat hidden state and input data
            - Wy and by: are for the output
        """
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Wh = np.concatenate((Whh, Whx), axis=0)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step

        Arg:
        - x_t is a np.ndarray of shape (m, i) with the input for the cell
        - m is the batche size for the data
        - h_prev: np.ndarray of shape (m, h) with the previous hidden state

        Returns: h_next, y
        - h_next: is the next hidden state
        - y: is the output of the cell
        """
        h_concat_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        h_next = np.tanh((np.matmul(h_concat_x.T, self.Wh)) + self.bh)
        y = (np.matmul(h_next, self.Wy)) + self.by

        return h_next, y
