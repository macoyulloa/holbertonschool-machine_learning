#!/usr/bin/env python3
"""Deep Convolutional Architectures"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """inception block
    Arg:
       A_prev: output from the previus layer
       filters: tuple or list containing the filter
                F1, F3R, F3,F5R, F5, FPP
    Returns: concatenated output
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    activation = 'relu'
    k_init = K.initializers.he_normal(seed=None)

    layer_1 = K.layers.Conv2D(filters=F1, kernel_size=1,
                              padding='same',
                              activation=activation,
                              kernel_initializer=k_init)(A_prev)

    layer_2R = K.layers.Conv2D(filters=F3R, kernel_size=1,
                               padding='same',
                               activation=activation,
                               kernel_initializer=k_init)(A_prev)

    layer_2 = K.layers.Conv2D(filters=F3, kernel_size=3,
                              padding='same',
                              activation=activation,
                              kernel_initializer=k_init)(layer_2R)

    layer_3R = K.layers.Conv2D(filters=F5R, kernel_size=1,
                               padding='same',
                               activation=activation,
                               kernel_initializer=k_init)(A_prev)

    layer_3 = K.layers.Conv2D(filters=F5, kernel_size=5,
                              padding='same',
                              activation=activation,
                              kernel_initializer=k_init)(layer_3R)

    layer_pool = K.layers.MaxPooling2D(pool_size=[3, 3],
                                       strides=1,
                                       padding='same')(A_prev)

    layer_poolR = K.layers.Conv2D(filters=FPP, kernel_size=1,
                                  padding='same',
                                  activation=activation,
                                  kernel_initializer=k_init)(layer_pool)

    layers_list = [layer_1, layer_2, layer_3, layer_poolR]
    return K.layers.concatenate(layers_list)
