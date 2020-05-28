#!/usr/bin/env python3
"Probability: Hidden Markov Chain"

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """The most likely sequence of hidden states for a hidden markov model

    Arg:
        - Observation: np.ndarray of shape (T,) that contains the index
                        of the observation
                - T is the number of observations
        - Emission: np.ndarray of shape (N, M) containing the emission
                    probab of a specific observation given a hidden state
                - Emission[i, j] is the probability of observing
                    j given the hidden state i
                - N is the number of hidden states
                - M is the number of all possible observations
        - Transition: 2D np.ndarray of shape (N, N) containing the transition
                    probabilities
                - Transition[i, j] probability of transitioning from the
                    hidden state i to j
        - Initial: np.ndarray of shape (N, 1) proba of starting in a particular
                    hidden state

    Returns: path, P, or None, None on failure
        - path: list of length T containing the most likely sequence of
                hidden states
        - P: is the probability of obtaining the path sequence
    """
    T = Observation.shape[0]
    N, M = Emission.shape
    N1, N2 = Transition.shape
    N3 = Initial.shape[0]

    if ((len(Observation.shape)) != 1) or (type(Observation) != np.ndarray):
        return (None, None)
    if ((len(Emission.shape)) != 2) or (type(Emission) != np.ndarray):
        return (None, None)
    if ((len(Transition.shape)) != 2) or (N != N1) or (N != N2):
        return (None, None)
    if (N1 != N2) or (type(Transition) != np.ndarray):
        return (None, None)
    prob = np.ones((1, N1))
    if not (np.isclose((np.sum(Transition, axis=1)), prob)).all():
        return (None, None)
    if ((len(Initial.shape)) != 2) or (type(Initial) != np.ndarray):
        return (None, None)
    if (N != N3):
        return (None, None)

    F = np.zeros((N, T))
    path = []
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for n in range(N):
            F[n, t] = (F[:, t-1].dot(Transition[:, n])) * \
                Emission[n, Observation[t]]
    path = np.max(F, axis=0)
    P = np.sum(path)

    return (path, P)
