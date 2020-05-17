#!/usr/bin/env python3
"""Baye's Theorem practice"""

from scipy import math, special


def posterior(x, n, p1, p2):
    """posterior probability that the probability of developing
       severe side effects falls within a specific range given data

    Arg:
       - x is the number of patients that develop severe side effects
       - n is the total number of patients observed
       - p1 is the lower bound on the range
       - p2 is the upper bound on the range

    Returns:
       - posterior prob that p is within the range [p1, p2] given x and n
    """
    if not isinstance(n, (int, float)) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if not isinstance(x, (int, float)) or (x < 0):
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if (x > n):
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float):
        raise ValueError("{} must be a float in the range [0, 1]".format(p1))
    if not isinstance(p1, float):
        raise ValueError("{} must be a float in the range [0, 1]".format(p1))
    if (p1 > 1) or (p1 < 0):
        raise ValueError("{} must be a float in the range [0, 1]".format(p1))
    if not isinstance(p2, float):
        raise ValueError("{} must be a float in the range [0, 1]".format(p2))
    if (p2 > 1) or (p2 < 0):
        raise ValueError("{} must be a float in the range [0, 1]".format(p2))
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # num = (np.math.factorial(n))
    # den = (np.math.factorial(x) * np.math.factorial(n - x))
    # factorial = num / den
    # D = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))

    # intersection = D * Pr
    # marginal = np.sum(intersection)
    # posterior = intersection / marginal

    return 1
