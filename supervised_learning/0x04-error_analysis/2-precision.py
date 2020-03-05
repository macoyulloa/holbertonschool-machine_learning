#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def precision(confusion):
    """ Calculate the precision of each class
    return the precision of each class
    """
    precision_list = []
    predic_cond_pos = np.sum(confusion, axis=0)
    for i in range(len(confusion)):
        for j in range(len(confusion)):
            if i == j:
                precision_list.append(confusion[i][j]/predic_cond_pos[i])
    return (precision_list)
