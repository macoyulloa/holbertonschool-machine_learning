#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """gradient descent RMSProp optimization algorithm"""
    Sd = (beta2 * s) + ((1 - beta2) * grad**2)
    W = var - (alpha * (grad / np.sqrt(Sd + epsilon)))
    return (W, Sd)
