#!/usr/bin/env python3
"""Gaited Recurrent Unit"""

import numpy as np


class GRUCell:
    """CRUCell class, vanila model"""

    def __init__(self, i, h, o):
        """ initialized the variables

        Arg:
        - i: is the dimensionality of the data
        - h: is the dimensionality of the hidden state
        - o: is the dimensionality of the outputs

        Public instance attributes: represent the weights and biases
            Wz, Wr, Wh, Wy, bz, br, bh
        - Wz and bz: are for the update gate
        - Wr and br: are for the reset gate
        - Wh and bh: are for the intermediate hidden state
        - Wy and by: are for the output
        """
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Wz = np.concatenate((Whh, Whx), axis=0)
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Wr = np.concatenate((Whh, Whx), axis=0)
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Wh = np.concatenate((Whh, Whx), axis=0)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """returns a sigmoid activation of the x array
        Arg:
        x: np.ndarray
        """
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / e_x.sum(axis=-1, keepdims=True)

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
        r = self.sigmoid(((np.matmul(h_concat_x.T, self.Wr)) + self.br))
        updated = self.sigmoid(((np.matmul(h_concat_x.T, self.Wz)) + self.bz))
        r_h_concat_x = np.concatenate(((r * h_prev).T, x_t.T), axis=0)
        h_candidate = np.tanh((np.matmul(r_h_concat_x.T, self.Wh)) + self.bh)
        h_next = updated * h_candidate + (1 - updated) * h_prev
        y = self.softmax((np.matmul(h_next, self.Wy)) + self.by)

        return h_next, y
