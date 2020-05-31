#!/usr/bin/env python3
"Probability: Hidden Markov Chain"

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model:

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

    Returns: (P, F), or (None, None) on failure
        - P: is the likelihood of the observations given the model
        - F: is a numpy.ndarray of shape (N, T) containing the forward
            path probabilities
            - F[i, j]: prob of being in hidden state i at time j given
                        the previous observations
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = (F[:, t - 1].dot(Transition[:, :])) * \
            Emission[:, Observation[t]]
    P = np.sum(F[:, -1])

    return (P, F)


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

    B = np.zeros((N, T))
    B[:, T - 1] += 1
    for t in range(T - 2, -1, -1):
        B[:, t] = (B[:, t + 1] * (Transition[:, :])
                   ).dot(Emission[:, Observation[t + 1]])

    P = np.sum(B[:, 0] * Initial.T * Emission[:, Observation[0]])

    return (P, B)


def baum_welch(Observations, N, M, Transition=None,
               Emission=None, Initial=None):
    """ performs the Baum-Welch algorithm for a hidden markov model:

    Arg:
        - Observation: np.ndarray of shape (T,) that contains the index
                        of the observation
                - T is the number of observations
        - N is the number of hidden states
        - M is the number of possible observations


    Returns: (Transition, Emission), or (None, None) on failure
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
    """
    T = Observations.shape[0]
    if ((len(Observations.shape)) != 1) or (type(Observations) != np.ndarray):
        return (None, None)
    if not isinstance(N, int) or N < 0:
        return (None, None)
    if not isinstance(M, int) or N < 0:
        return (None, None)

    if Transition is None:
        Transition = np.random.uniform(0, 1, size=(N, N))
    if Emission is None:
        Emission = np.random.uniform(0, 1, size=(N, M))
    if Initial is None:
        # Initial = np.repeat((1/N, 1), N)
        Initial = np.random.uniform(0, 1, size=(N, 1))

    print(Transition)
    print(Emission)
    print(Initial)
    N, M = Emission.shape
    N1, N2 = Transition.shape
    N3 = Initial.shape[0]
    if ((len(Emission.shape)) != 2) or (type(Emission) != np.ndarray):
        return (None, None)
    if ((len(Transition.shape)) != 2) or (N != N1) or (N != N2):
        return (None, None)
    if (N1 != N2) or (type(Transition) != np.ndarray):
        return (None, None)
    # prob = np.ones((1, N1))
    # if not (np.isclose((np.sum(Transition, axis=1)), prob)).all():
    #    return (None, None)
    if ((len(Initial.shape)) != 2) or (type(Initial) != np.ndarray):
        return (None, None)
    if (N != N3):
        return (None, None)

    V = Observations
    b = Emission
    a = Transition
    M = N
    while True:
        # forward
        Prob_forward, alpha = forward(Observations, Emission,
                                      Transition, Initial)
        alpha = alpha.T
        # backward
        Prob_backward, beta = backward(Observations, Emission,
                                       Transition, Initial)
        beta = beta.T
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(
                np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * \
                    b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

        if np.isclose(Prob_forward, Prob_backward):
            break

    return (Transition, Emission)
