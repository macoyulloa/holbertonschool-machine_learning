#!/usr/bin/env python3
"""t-SNE method"""

import numpy as np


def HP(Di, beta):
    """calculates the Shannon entropy & P affinities relative to a data point:
    Arg:
      - Di: np.ndarray (n - 1,) pariwise distances between a data point
           and all other points
         n is the number of data points
      - beta: is the beta value for the Gaussian distribution

    Returns: (Hi, Pi)
      - Hi: is the Shannon entropy of the points
      - Pi: np.ndarray (n - 1,) contain the P affinities of the points
    """
    Pi = (np.exp(-Di * beta)) / (np.sum(np.exp(-Di * beta)))
    Hi = - np.sum(Pi * np.log2(Pi))

    return (Hi, Pi)
