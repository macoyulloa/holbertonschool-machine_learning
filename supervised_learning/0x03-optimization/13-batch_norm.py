#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """batch normalization"""
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)

    std = np.sqrt(var + epsilon)
    Z_centered = Z - mean
    Z_norm = Z_centered / std
    Z_n_batch = (gamma * Z_norm) + beta
    return Z_n_batch
