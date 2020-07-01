#!/usr/bin/env python3
"""Recurrent Neural Network"""

import numpy as np


class BidirectionalCell:
    """Bidimensional cell Model of a RNN"""

    def __init__(self, i, h, o):
        """ initialized the variables

        Arg:
        - i: is the dimensionality of the data
        - h: is the dimensionality of the hidden state
        - o: is the dimensionality of the outputs

        Public instance attributes Wh, Wy, bh, by that represent
            the weights and biases of the cell
            - Whf and bhf: hidden states in the forward direction
            - Whb and bhb: hidden states in the backward direction
            - Wy and by: are for the outputs
        """
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Whf = np.concatenate((Whh, Whx), axis=0)
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Whb = np.concatenate((Whh, Whx), axis=0)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step

        Arg:
        - x_t is a np.ndarray of shape (m, i) with the input for the cell
            - m is the batche size for the data
        - h_prev: np.ndarray of shape (m, h) with the previous hidden state

        Returns: h_next
        - h_next: is the next hidden state
        """
        h_concat_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        h_next = np.tanh((np.matmul(h_concat_x.T, self.Whf)) + self.bhf)
        # y = self.softmax((np.matmul(h_next, self.Wy)) + self.by)

        return h_next

    def backward(self, h_next, x_t):
        """ calculates the hidden state in the backward direction

        Arg:
        - x_t: np.ndarray of shape (m, i) with the data input for the cell
                m is the batch size for the data
        - h_next: np.ndarray of shape (m, h) with the next hidden state

        Returns: h_pev, the previous hidden state
        """
        h_concat_x = np.concatenate((h_next.T, x_t.T), axis=0)
        h_prev = np.tanh((np.matmul(h_concat_x.T, self.Whb)) + self.bhb)

        return h_prev
