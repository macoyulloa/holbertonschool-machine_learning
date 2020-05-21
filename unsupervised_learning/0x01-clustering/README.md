0x01. Clustering
================

Tasks
-----

#### 0\. Initialize K-means
Function `def initialize(X, k):` that initializes cluster centroids for K-means

#### 1\. K-means

Function `def kmeans(X, k, iterations=1000):` that performs K-means on a dataset.

#### 2\. Variance

Function `def variance(X, C):` that calculates the total intra-cluster variance for a data set:

#### 3\. Optimize k mandatory

Function`def optimum_k(X, kmin=1, kmax=None, iterations=1000):` that tests for the optimum number of clusters by variance:

#### 4\. Initialize GMM

Function `def initialize(X, k):` that initializes variables for a Gaussian Mixture Model:

#### 5\. PDF

function `def pdf(X, m, S):` that calculates the probability density function of a Gaussian distribution.

#### 6\. Expectation

Function `def expectation(X, pi, m, S):` that calculates the expectation step in the EM algorithm for a GMM

#### 7\. Maximization

Function `def maximization(X, g):` that calculates the maximization step in the EM algorithm for a GMM

#### 8\. EM

Function `def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):` that performs the expectation maximization for a GMM

#### 9\. BIC

Function `def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):` that finds the best number of clusters for a GMM using the Bayesian Information Criterion

#### 10\. Hello, sklearn!

Function `def kmeans(X, k):` that performs K-means on a dataset

#### 11\. GMM

Function `def gmm(X, k):` that calculates a GMM from a dataset

#### 12\. Agglomerative

Function `def agglomerative(X, dist):` that performs agglomerative clustering on a dataset
