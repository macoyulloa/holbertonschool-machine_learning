#!/usr/bin/env python3
"""Baye's Theorem practice"""

import numpy as np


def intersection(x, n, P, Pr):
    """ intersection of obtaining this data with the various
        hypothetical probabilities
    Arg:
       - x is the number of patients that develop severe side effects
       - n is the total number of patients observed
       - P is a 1D numpy.ndarray containing the various hypothetical
           probabilities of developing severe side effects
       - Pr is a 1D numpy.ndarray containing the prior beliefs of P

    Returns: D np.ndarray
       - D: with the likelihood of obtaining the intersection of
            obtaining x and n, for each probability in P
    """
    if not isinstance(n, (int, float)) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if not isinstance(x, (int, float)) or (x < 0):
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if (x > n):
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or (P.shape != Pr.shape):
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    suma = (np.sum(Pr))
    if not (np.isclose(suma, 1)):
        raise ValueError("Pr must sum to 1")

    num = (np.math.factorial(n))
    den = (np.math.factorial(x) * np.math.factorial(n - x))
    factorial = num / den
    D = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))

    intersection = D * Pr

    return intersection
