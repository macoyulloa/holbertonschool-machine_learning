#!/usr/bin/env python3
"Probability: Markov Chain"

import numpy as np


def markov_chain(P, s, t=1):
    """ determines the prob of a markov chain being in a particular state
        after a specified number of iterations:

    Arg:
        - P: square 2D np.ndarray of shape (n, n) representing the
            transition matrix
        - P[i, j]: is the probability of transitioning from state i to state j
        - n: is the number of states in the markov chain
        - s: is a numpy.ndarray of shape (1, n) representing the probability
            of starting in each state
        - t: is the number of iterations that the markov chain has been through

    Returns: np.ndarray of shape (1, n)
        - representing the probability of being in a specific state after t
            iterations, or None on failure
    """
    n1, n2 = P.shape
    if ((len(P.shape)) != 2):
        return (None)
    if (n1 != n2) or (type(P) != np.ndarray):
        return (None)
    prob = np.ones((1, n1))
    if not (np.isclose((np.sum(P, axis=1)), prob)).all():
        return (None)
    if (n1 != s.shape[1]) or (s.shape[0] != 1):
        return (None)
    if not np.isclose((np.sum(s)), 1):
        return (None)
    if not isinstance(t, int) or (t < 0):
        return (None)

    for i in range(t):
        s = np.matmul(s, P)

    return (s)
