#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def normalization_constants(X):
    """calculates the normalization"""
    mean = np.mean(X, axis=0)
    standard_desv = np.std(X, axis=0)
    return (mean, standard_desv)
