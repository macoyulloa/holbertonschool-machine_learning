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

    def forward_prop(self, X):
        """forward propagation of the neural network"""
        matrix_mul1 = (np.matmul(self.__W1, X)) + self.__b1
        self.__A1 = 1 / (1 + np.exp(- matrix_mul1))
        matrix_mul2 = (np.matmul(self.__W2, self.__A1)) + self.__b2
        self.__A2 = 1 / (1 + np.exp(- matrix_mul2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """evaluates the neuronal network's predictions """
        self.__A = self.forward_prop(X)
        evalu = np.where(self.__A2 >= 0.5, 1, 0)
        cost_y = self.cost(Y, self.__A2)
        return (evalu, cost_y)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ gradient descent on the neural netwok """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1/m) * np.matmul(A1, dz2.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

        dz11 = np.matmul(self.__W2.T, dz2)
        dz12 = A1 * (1 - A1)
        dz1 = dz11 * dz12
        dw1 = (1/m) * np.matmul(dz1, X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2).T
        self.__b2 = self.__b2 - (alpha * db2)
