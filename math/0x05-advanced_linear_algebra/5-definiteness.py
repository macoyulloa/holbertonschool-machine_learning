#!/usr/bin/env python3
""" Linear and Matricial Algebra"""

import numpy as np


def definiteness(matrix):
    """ Calculates the definitness of a matrix
    Arg:
       - matrix: np.array whose definiteness should be calculated

    Returns: a string
       - Positive definite
       - Positive semi-definite
       - Negative semi-definite
       - Negative definite
       - Indefinite
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.linalg.eig(matrix):
        return None

    w, v = np.linalg.eig(matrix)

    if np.all(w == 0):
        return None
    if np.all(w > 0):
        return "Positive definite"
    if np.all(w >= 0):
        return "Positive semi-definite"
    if np.all(w < 0):
        return "Negative definite"
    if np.all(w <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
