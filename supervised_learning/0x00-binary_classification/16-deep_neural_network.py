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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(self.L):
            if layers[l] <= 0 or type(layers[l]) is not int:
                raise TypeError("layers must be a list of positive integers")
            if l is 0:
                he_init = np.random.randn(layers[l], nx)*np.sqrt(2/nx)
                self.weights['W' + str(l+1)] = he_init
            if l > 0:
                he_init1 = np.random.randn(layers[l], layers[l-1])
                he_init2 = np.sqrt(2/layers[l-1])
                self.weights['W' + str(l+1)] = he_init1 * he_init2
            self.weights['b' + str(l + 1)] = np.zeros((layers[l], 1))
