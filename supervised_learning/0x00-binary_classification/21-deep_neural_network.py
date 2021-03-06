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

    def cost(self, Y, A):
        """Cost calculation using logistic regression"""
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """evaluates the neuronal network predictions"""
        A, self.__cache = self.forward_prop(X)
        evalu = np.where(A >= 0.5, 1, 0)
        cost_y = self.cost(Y, A)
        return (evalu, cost_y)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """one pass of gradient descent on the neural network"""
        weights1 = self.__weights.copy()
        m = Y.shape[1]
        A3 = self.__cache['A'+str(self.__L)]
        A2 = self.__cache['A'+str(self.__L - 1)]
        W3 = weights1['W'+str(self.__L)]
        b3 = weights1['b'+str(self.__L)]
        dz_list = {}
        dz3 = A3 - Y
        dz_list['dz'+str(self.__L)] = dz3
        dw3 = (1/m) * np.matmul(A2, dz3.T)
        db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)
        self.__weights['W'+str(self.__L)] = W3 - (alpha * dw3).T
        self.__weights['b'+str(self.__L)] = b3 - (alpha * db3)

        for l in range(self.__L - 1, 0, -1):
            A_curr = self.__cache['A'+str(l)]
            A_bef = self.__cache['A'+str(l-1)]
            W_curr = weights1['W'+str(l)]
            W_next = weights1['W'+str(l+1)]
            b_curr = weights1['b'+str(l)]
            dz1 = np.matmul(W_next.T, dz_list['dz'+str(l+1)])
            dz2 = A_curr * (1 - A_curr)
            dz = dz1 * dz2
            dw = (1/m) * np.matmul(A_bef, dz.T)
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            dz_list['dz'+str(l)] = dz
            self.__weights['W'+str(l)] = W_curr - (alpha * dw).T
            self.__weights['b'+str(l)] = b_curr - (alpha * db)
