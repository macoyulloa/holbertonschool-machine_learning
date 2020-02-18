#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork():
    """deepNeuronalNetwork performing binary classification"""

    def __init__(self, nx, layers):
        """ defining, initialized the variables """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) is 0:
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for l in range(self.__L):
            if layers[l] <= 0 or type(layers[l]) is not int:
                raise TypeError("layers must be a list of positive integers")
            if l is 0:
                he_init = np.random.randn(layers[l], nx)*np.sqrt(2/nx)
                self.__weights['W' + str(l+1)] = he_init
            if l > 0:
                he_init1 = np.random.randn(layers[l], layers[l-1])
                he_init2 = np.sqrt(2/layers[l-1])
                self.__weights['W' + str(l+1)] = he_init1 * he_init2
            self.__weights['b' + str(l + 1)] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """Getter for length of the layers"""
        return self.__L

    @property
    def cache(self):
        """Get the dict with all A of the hidden leyers"""
        return self.__cache

    @property
    def weights(self):
        """Get a dict with all the weight and the bias of the layers"""
        return self.__weights

    def forward_prop(self, X):
        """forward propagation of the deep Neural Network"""
        for l in range(self.__L + 1):
            if l is 0:
                self.__cache['A' + str(l)] = X
            else:
                W_curr = self.__weights['W'+str(l)]
                b_curr = self.__weights['b'+str(l)]
                A_prev = self.__cache['A'+str(l-1)]
                matrix_mul = (np.matmul(W_curr, A_prev)) + b_curr
                self.__cache['A' + str(l)] = 1 / (1 + np.exp(- matrix_mul))
        return (self.__cache['A'+str(self.__L)], self.__cache)
