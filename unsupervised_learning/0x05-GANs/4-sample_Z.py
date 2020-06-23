#!/usr/bin/env python3
"""Generative Adversarial Networks"""

import numpy as np


def sample_Z(m, n):
    """creates input for the generator:

    Arg:
        - m is the number of samples that should be generated
        - n is the number of dimensions of each sample

    All samples should be taken from a random uniform distribution within
    the range [-1, 1]

    Returns:
        - Z, a numpy.ndarray of shape (m, n) with the uniform samples
    """
    Z = np.random.uniform(-1., 1., size=(m, n))
    return Z
