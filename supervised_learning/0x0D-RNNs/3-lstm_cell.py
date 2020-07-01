#!/usr/bin/env python3
"""Long Short Term Memory model"""

import numpy as np


class LSTMCell:
    """LSTMCell class model"""

    def __init__(self, i, h, o):
        """ initialized the variables

        Arg:
        - i: is the dimensionality of the data
        - h: is the dimensionality of the hidden state
        - o: is the dimensionality of the outputs

        Public instance attributes: represent the weights and biases
            Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
        - Wf and bf are for the forget gate
        - Wu and bu are for the update gate
        - Wc and bc are for the intermediate cell state
        - Wo and bo are for the output gate
        - Wy and by are for the outputs
        """
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Wf = np.concatenate((Whh, Whx), axis=0)
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Wu = np.concatenate((Whh, Whx), axis=0)
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Wc = np.concatenate((Whh, Whx), axis=0)
        Whh = np.random.randn(h, h)
        Whx = np.random.randn(i, h)
        self.Wo = np.concatenate((Whh, Whx), axis=0)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """ performs forward propagation for one time step

        Arg:
        - x_t is a np.ndarray of shape (m, i) with the input for the cell
        - m is the batche size for the data
        - h_prev: np.ndarray of shape (m, h) with the previous hidden state
        - c_prev is a np.ndarray of shape (m, h) with the previous cell state

        Returns: h_next, c_next, y
        - h_next: is the next hidden state
        - c_next is the next cell state
        - y: is the output of the cell
        """
        h_concat_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        forget = self.sigmoid(((np.matmul(h_concat_x.T, self.Wf)) + self.bf))
        updated = self.sigmoid(((np.matmul(h_concat_x.T, self.Wu)) + self.bu))
        output = self.sigmoid(((np.matmul(h_concat_x.T, self.Wo)) + self.bo))
        c_candidate = np.tanh((np.matmul(h_concat_x.T, self.Wc)) + self.bc)
        c_next = updated * c_candidate + forget * c_prev
        h_next = output * c_next
        y = self.softmax((np.matmul(h_next, self.Wy)) + self.by)

        return h_next, c_next, y
