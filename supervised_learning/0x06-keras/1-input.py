#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: number of features
    layers: list with the nodes per layer
    activations: active function per layer
    lambtha: l2 rate 0.001
    keep_prob: to keep the node for dropout
    """
    inputs = tf.keras.Input(shape=(nx,))
    r_l2 = tf.keras.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            output = tf.keras.layers.Dense(layers[i],
                                           activation=activations[i],
                                           kernel_regularizer=r_l2)(inputs)
        else:
            dropout = tf.keras.layers.Dropout(keep_prob)(output)
            output = tf.keras.layers.Dense(layers[i],
                                           activation=activations[i],
                                           kernel_regularizer=r_l2)(dropout)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model
