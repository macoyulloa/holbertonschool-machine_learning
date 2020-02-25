#!/usr/bin/env python3
"""forward prop of the model"""

import tensorflow as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward propagation"""
    l = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        ln = create_layer(l, layer_sizes[i], activations[i])
        l = ln
    return l
