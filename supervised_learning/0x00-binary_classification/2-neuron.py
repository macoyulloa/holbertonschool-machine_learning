#!/usr/bin/env python3
" defines a single neuron performing binary classification "

import numpy as np


class Neuron():
    " single neuron binary classification "

    def __init__(self, nx):
        " initialized variables "
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        " gett the weight as W "
        return self.__W

    @property
    def b(self):
        " gett the bias as b"
        return self.__b

    @property
    def A(self):
        " get the activation function as A"
        return self.__A

    def forward_prop(self, X):
        " return the forward propagation of the neuron, using sigmoid "
        matrix_mul = (np.matmul(self.__W, X)) + self.__b
        self.__A = 1 / (1 + np.exp(- matrix_mul))
        return self.__A
