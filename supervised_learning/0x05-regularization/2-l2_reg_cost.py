#!/usr/bin/env python3
"""regularization of a model"""

import tensorflow as tf


def l2_reg_cost(cost):
    """calculates the cost of a neural network L2
    """
    return cost + tf.losses.get_regularization_losses()
