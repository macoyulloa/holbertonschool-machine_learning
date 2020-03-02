#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def moving_average(data, beta):
    """exponential weighted moving average of a data set"""
    EMA = []
    Vt = 0
    for i in range(len(data)):
        Vt = (beta * Vt) + ((1 - beta) * data[i])
        bias_correct = 1 - (beta ** (i + 1))
        Vt_corrected = Vt / bias_correct
        EMA.append(Vt_corrected)
    return EMA
