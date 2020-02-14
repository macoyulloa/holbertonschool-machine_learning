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

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
