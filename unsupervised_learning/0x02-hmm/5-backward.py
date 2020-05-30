#!/usr/bin/env python3
"Probability: Hidden Markov Chain"

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """that performs the backward algorithm for a hidden markov model

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

    Returns: (P, B), or (None, None) on failure
        - P: is the likelihood of the observations given the model
        - B. np.ndarray of shape (N, T) containing the backward path probs
            - B[i, j]: is the probability of generating the future
                        observations from hidden state i at time j
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

    B = np.zeros((N, T))
    B[:, T - 1] += 1
    print(B[:, T-1])

    # for t in range(T - 2, -1, -1):
    #    for n in range(N):
    #        B[n, t] = (B[:, t + 1] * Emission[:, Observation[t + 1]]
    #                   ).dot((Transition[:, n]))

    for t in range(T - 2, -1, -1):
        B[:, t] = (B[:, t + 1] * (Transition[:, :])
                   ).dot(Emission[:, Observation[t + 1]])

    # F[:, 0] = Initial.T * Emission[:, Observation[0]]
    B[:, 0] = B[:, 0] * Initial.T * Emission[:, Observation[0]]

    P = np.sum(B[:, 0])

    return (P, B)
