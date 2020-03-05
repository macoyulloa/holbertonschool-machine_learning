#!/usr/bin/env python3
"""create a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """ Calculate the sensitive of each class
    return the sensitivy of each class
    """
    sensitivity_list = []
    cond_positive = np.sum(confusion, axis=1)
    for i in range(len(confusion)):
        for j in range(len(confusion)):
            if i == j:
                sensitivity_list.append(confusion[i][j]/cond_positive[i])
    return (sensitivity_list)
