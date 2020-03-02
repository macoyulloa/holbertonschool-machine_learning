#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """gradient descent with momentum optimization algorithm"""
    Vd = (beta1 * v) + ((1 - beta1) * grad)
    W = var-(alpha * Vd)
    return (W, Vd)
