#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ Calculate the f1_score of each class
    return the f1_score of each class
    """
    s = sensitivity(confusion)
    p = precision(confusion)
    f1_score = []
    for i in range(len(s)):
        f1_score.append(2 * ((p[i] * s[i])/(p[i] + s[i])))
    return (f1_score)
