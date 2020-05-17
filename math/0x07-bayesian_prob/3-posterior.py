#!/usr/bin/env python3
"""Baye's Theorem practice"""

import numpy as np


def posterior(x, n, P, Pr):
    """posterior probability for the various hypothetical
       probabilities of developing severe side effects given the data:

    Arg:
       - x is the number of patients that develop severe side effects
       - n is the total number of patients observed
       - P is a 1D numpy.ndarray containing the various hypothetical
           probabilities of developing severe side effects
       - Pr is a 1D numpy.ndarray containing the prior beliefs of P

    Returns:
       - posterior probability of each probability in P given x and n
    """
    if not isinstance(n, int) or (n < 0):
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or (x <= 0):
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if (x > n):
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or (P.shape != Pr.shape):
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if (np.any((np.vectorize(lambda x: 0 <= x <= 1)(P)) is False)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if (np.any((np.vectorize(lambda x: 0 <= x <= 1)(Pr)) is False)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    suma = (np.sum(Pr))
    if (np.isclose(suma, 1)) is False:
        raise ValueError("Pr must sum to 1")

    num = (np.math.factorial(n))
    den = (np.math.factorial(x) * np.math.factorial(n - x))
    factorial = num / den
    D = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))

    intersection = D * Pr
    marginal = np.sum(intersection)
    posterior = intersection / marginal

    return posterior
