#!/usr/bin/env python3
"Probability: Markov Chain"

import numpy as np


def regular(P):
    """ determines the steady state probabilities of a regular markov chain

    Arg:
        - P: square 2D np.ndarray of shape (n, n) representing the
            transition matrix
        - P[i, j]: is the probability of transitioning from state i to state j
        - n: is the number of states in the markov chain

    Returns: np.ndarray of shape (1, n) containing the steady state
            probabilities, or None on failure
    """
    n1, n2 = P.shape
    if (len(P.shape) != 2):
        return (None)
    if (n1 != n2) or (type(P) != np.ndarray):
        return (None)
    prob = np.ones((1, n1))
    if not (np.isclose((np.sum(P, axis=1)), prob)).all():
        return (None)

    s = np.ones((1, n1)) / n1
    P_pow = P.copy()
    while True:
        s_prev = s
        s = np.matmul(s, P)
        P_pow = P * P_pow
        if np.any(P_pow <= 0):
            return (None)
        if np.all(s_prev == s):
            return (s)
