#!/usr/bin/env python3
"""create a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """ Creates a confusion matrix.
    return a confusion np array of shappe classes, classes
    """
    return np.matmul(labels.T, logits)
