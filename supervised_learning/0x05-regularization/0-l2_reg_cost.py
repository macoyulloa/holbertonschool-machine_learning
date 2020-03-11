#!/usr/bin/env python3
"""regularization of a model"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """cost of a neural network with L2 regularization
    cost: cost of the network without L2
    lambtha: regularization param
    weights: dic with w and b
    L: num of layers
    m: num of data points
    Return:  cost of the network accounting for L2 regularization
    """
    weight_sum = 0
    for l in range(1, L + 1):
        weight_sum += np.linalg.norm(weights['W'+str(l)])
    return cost + ((lambtha/(2*m)) * weight_sum)
