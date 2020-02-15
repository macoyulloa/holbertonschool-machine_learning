#!/usr/bin/env python3
""" neural network with one hidden layer """

import numpy as np


class NeuralNetwork():
    """
    neural network with one hidden layer binary classification
    """

    def __init__(self, nx, nodes):
        """ Constructor method """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=((1, nodes)))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for W1 hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Getter method for b1 for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Getter A1 activated output for hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Getter W2 output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Getter b2 attribute, bias for neuron"""
        return self.__b2

    @property
    def A2(self):
        """Getter A2 attribute, activated output neuron"""
        return self.__A2
