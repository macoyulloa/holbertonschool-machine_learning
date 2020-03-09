#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def specificity(confusion):
    """ Calculate the specificity of each class
    return the specifity of each class
    """
    true_pos = np.diagonal(confusion)
    m_total = np.sum(confusion)
    array_m_total = np.full_like(confusion[0], m_total)
    cross1 = np.sum(confusion, axis=0)
    cross2 = np.sum(confusion.T, axis=0)
    true_neg = array_m_total + true_pos - cross1 - cross2
    fp = cross1 - true_pos
    return true_neg / (fp + true_neg)
