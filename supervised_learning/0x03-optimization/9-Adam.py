#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Adam optimization algorithm"""
    Vd = (beta1 * v) + ((1 - beta1) * grad)
    Vd_correct = Vd / (1 - beta1**t)
    Sd = (beta2 * s) + ((1 - beta2) * grad**2)
    Sd_correct = Sd / (1 - beta2**t)
    v_updated = var - alpha * (Vd_correct/((Sd_correct**(1/2)) + epsilon))
    return (v_updated, Vd, Sd)
