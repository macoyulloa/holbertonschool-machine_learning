0x07. Bayesian Probability
==========================

Tasks
-----

#### 0\. Likelihood

You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, `n` patients take the drug and `x` patients develop severe side effects. You can assume that `x` follows a binomial distribution.

`def likelihood(x, n, P):` that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects


#### 1\. Intersection

Based on `0-likelihood.py`, adding the function `def intersection(x, n, P, Pr):` that calculates the intersection of obtaining this data with the various hypothetical probabilities


#### 2\. Marginal Probability

Based on `1-intersection.py`, write a function `def marginal(x, n, P, Pr):` that calculates the marginal probability of obtaining the data.


#### 3\. Posterior

Based on `2-marginal.py`, function `def posterior(x, n, P, Pr):` that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data.


#### 4\. Continuous Posterior

Based on `3-posterior.py`, write a function `def posterior(x, n, p1, p2):` that calculates the posterior probability that the probability of developing severe side effects falls within a specific range given the data.
