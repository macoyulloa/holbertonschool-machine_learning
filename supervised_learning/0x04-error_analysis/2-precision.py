#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def precision(confusion):
    """ Calculate the precision of each class
    return the precision of each class
    """
    predic_cond_pos = np.sum(confusion, axis=0)
    true_pos = np.diagonal(confusion)
    precision = true_pos / predic_cond_pos
    return (precision)
