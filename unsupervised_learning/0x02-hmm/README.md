0x02. Hidden Markov Models
==========================


Tasks
-----

#### 0\. Markov Chain

Function `def markov_chain(P, s, t=1):` that determines the probability of a markov chain being in a particular state after a specified number of iterations:

-   `P` is a square 2D `numpy.ndarray` of shape `(n, n)` representing the transition matrix
    -   `P[i, j]` is the probability of transitioning from state `i` to state `j`
    -   `n` is the number of states in the markov chain
-   `s` is a `numpy.ndarray` of shape `(1, n)` representing the probability of starting in each state
-   `t` is the number of iterations that the markov chain has been through
-   Returns: a `numpy.ndarray` of shape `(1, n)` representing the probability of being in a specific state after `t` iterations, or `None` on failure

#### 1\. Regular Chains

Function `def regular(P):` that determines the steady state probabilities of a regular markov chain:

-   `P` is a is a square 2D `numpy.ndarray` of shape `(n, n)` representing the transition matrix
    -   `P[i, j]` is the probability of transitioning from state `i` to state `j`
    -   `n` is the number of states in the markov chain
-   Returns: a `numpy.ndarray` of shape `(1, n)` containing the steady state probabilities, or `None` on failure

#### 2\. Absorbing Chains

Function `def absorbing(P):` that determines if a markov chain is absorbing:

-   P is a is a square 2D `numpy.ndarray` of shape `(n, n)` representing the transition matrix
    -   `P[i, j]` is the probability of transitioning from state `i` to state `j`
    -   `n` is the number of states in the markov chain
-   Returns: `True` if it is absorbing, or `False` on failure

#### 3\. The Forward Algorithm

Function `def forward(Observation, Emission, Transition, Initial):` that performs the forward algorithm for a hidden markov model:

-   `Observation` is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation
    -   `T` is the number of observations
-   `Emission` is a `numpy.ndarray` of shape `(N, M)` containing the emission probability of a specific observation given a hidden state
    -   `Emission[i, j]` is the probability of observing `j` given the hidden state `i`
    -   `N` is the number of hidden states
    -   `M` is the number of all possible observations
-   `Transition` is a 2D `numpy.ndarray` of shape `(N, N)` containing the transition probabilities
    -   `Transition[i, j]` is the probability of transitioning from the hidden state `i` to `j`
-   `Initial` a `numpy.ndarray` of shape `(N, 1)` containing the probability of starting in a particular hidden state
-   Returns: `P, F`, or `None, None` on failure
    -   `P` is the likelihood of the observations given the model
    -   `F` is a `numpy.ndarray` of shape `(N, T)` containing the forward path probabilities
        -   `F[i, j]` is the probability of being in hidden state `i` at time `j` given the previous observations


#### 4\. The Viretbi Algorithm

Function `def viterbi(Observation, Emission, Transition, Initial):` that calculates the most likely sequence of hidden states for a hidden markov model:

-   `Observation` is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation
    -   `T` is the number of observations
-   `Emission` is a `numpy.ndarray` of shape `(N, M)` containing the emission probability of a specific observation given a hidden state
    -   `Emission[i, j]` is the probability of observing `j` given the hidden state `i`
    -   `N` is the number of hidden states
    -   `M` is the number of all possible observations
-   `Transition` is a 2D `numpy.ndarray` of shape `(N, N)` containing the transition probabilities
    -   `Transition[i, j]` is the probability of transitioning from the hidden state `i` to `j`
-   `Initial` a `numpy.ndarray` of shape `(N, 1)` containing the probability of starting in a particular hidden state
-   Returns: `path, P`, or `None, None` on failure
    -   `path` is the a list of length `T` containing the most likely sequence of hidden states
    -   `P` is the probability of obtaining the `path` sequence

#### 5\. The Backward Algorithm

Function `def backward(Observation, Emission, Transition, Initial):` that performs the backward algorithm for a hidden markov model:

-   `Observation` is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation
    -   `T` is the number of observations
-   `Emission` is a `numpy.ndarray` of shape `(N, M)` containing the emission probability of a specific observation given a hidden state
    -   `Emission[i, j]` is the probability of observing `j` given the hidden state `i`
    -   `N` is the number of hidden states
    -   `M` is the number of all possible observations
-   `Transition` is a 2D `numpy.ndarray` of shape `(N, N)` containing the transition probabilities
    -   `Transition[i, j]` is the probability of transitioning from the hidden state `i` to `j`
-   `Initial` a `numpy.ndarray` of shape `(N, 1)` containing the probability of starting in a particular hidden state
-   Returns: `P, B`, or `None, None` on failure
    -   `P`is the likelihood of the observations given the model
    -   `B` is a `numpy.ndarray` of shape `(N, T)` containing the backward path probabilities
        -   `B[i, j]` is the probability of generating the future observations from hidden state `i` at time `j`

#### 6\. The Baum-Welch Algorithm

Function `def baum_welch(Observations, N, M, Transition=None, Emission=None, Initial=None):` that performs the Baum-Welch algorithm for a hidden markov model:

-   `Observation` is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation
    -   `T` is the number of observations
-   `N` is the number of hidden states
-   `M` is the number of possible observations
-   `Transition` is the initialized transition probabilities, defaulted to `None`
-   `Emission` is the initialized emission probabilities, defaulted to `None`
-   `Initial` is the initiallized starting probabilities, defaulted to `None`
-   If `Transition`, `Emission`, or `Initial` is None, initialize the probabilities as being a uniform distribution
-   Returns: the converged `Transition, Emission`, or `None, None` on failure
