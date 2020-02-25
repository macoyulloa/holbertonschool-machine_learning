#!/usr/bin/env python3
"""traning the model that buils, trains, and saves"""

import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """saving the neural network"""

