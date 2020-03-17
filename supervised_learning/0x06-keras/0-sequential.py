#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: number of features
    layers: list with the nodes per layer
    activations: active function per layer
    lambtha: l2 rate 0.001
    keep_prob: to keep the node for dropout
    """

    model = K.Sequential()
    reg_l2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                                     activation=activations[i],
                                     kernel_regularizer=reg_l2,
                                     name='dense'))
        else:
            model.add(K.layers.Dropout(1-keep_prob))
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=reg_l2,
                                     name='dense_' + str(i)))
    return model
