#!/usr/bin/env python3
"""Early stopping regularization of a model"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Know if you should stop gradient descent
    cost: validation cost of the neural
    opt_cost: lowest recorded validation cost
    threshold: threshold used for stoping
    patience: used for early stopping
    count: how long the thredshold has not been met
    return: boolean
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if (count == patience):
        boolean = True
    else:
        boolean = False
    return (boolean, count)
