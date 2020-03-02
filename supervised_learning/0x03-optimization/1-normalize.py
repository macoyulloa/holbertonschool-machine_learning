#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np


def normalize(X, m, s):
    """normalizing the matrix"""
    return (X - m) / s
