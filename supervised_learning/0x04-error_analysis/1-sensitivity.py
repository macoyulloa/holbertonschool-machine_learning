#!/usr/bin/env python3
"""create a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """ Calculate the sensitive of each class
    return the sensitivy of each class
    """
    cond_positive = np.sum(confusion, axis=1)
    true_pos = np.diagonal(confusion)
    sensitivity = true_pos / cond_positive
    return (sensitivity)
