#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    network: is the model to optimize
    alpha: learning rate
    beta1: first Adam opti param
    beta2: second Adam opti param
    """
    adam_op = K.optimizers.Adam(lr=alpha,
                                beta_1=beta1,
                                beta_2=beta2)
    network.compile(optimizer=adam_op,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
