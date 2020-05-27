#!/usr/bin/env python3
"Probability: Markov Chain"

import numpy as np


def absorbing(P):
    """ determines if a markov chain is absorbing

    Arg:
        - P: square 2D np.ndarray of shape (n, n) representing the
            transition matrix
        - P[i, j]: is the probability of transitioning from state i to state j
        - n: is the number of states in the markov chain

    Returns: True if it is absorbing, or False on failure
    """
    n1, n2 = P.shape
    if (len(P.shape) != 2):
        return (None)
    if (n1 != n2) or (type(P) != np.ndarray):
        return (None)
    prob = np.ones((1, n1))
    if not (np.isclose((np.sum(P, axis=1)), prob)).all():
        return (None)

    if (np.all(np.diag(P) == 1)):
        return True
    if not np.any(np.diagonal(P) == 1):
        return False

    for i in range(n1):
        for j in range(n2):
            if (i == j) and (i + 1 < len(P)):
                if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                    return False
    return True
