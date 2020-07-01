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
        - h_t is the initial hidden state in the backward direction,
            given as a numpy.ndarray of shape (m, h)

        Returns: H, Y
        - H: np.ndarray containing all of the concat hidden states
        - Y: np.ndarray containing all of the outputs
    """
    