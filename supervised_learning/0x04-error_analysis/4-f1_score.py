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
    return (2 * ((p * s)/(p + s)))
