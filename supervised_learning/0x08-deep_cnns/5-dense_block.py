#!/usr/bin/env python3
"""Deep Convolutional Architectures"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """inception block
    Arg:
       -  X: is the output from the previous layer
       -  nb_filters: int the number of filters in X
       -  growth_rate: is the growth rate for the dense block
       -  layers: is the number of layers in the dense block
    Returns: concatenated output of each layer within the Dense
             Block and the num of f within the concatenated outputs
    """
    k_init = K.initializers.he_normal(seed=None)

    for layer in range(layers):
        batch_norm1 = K.layers.BatchNormalization()(X)
        activation1 = K.layers.Activation('relu')(batch_norm1)
        f_1x1 = K.layers.Conv2D(filters=128,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=k_init)(activation1)
        batch_norm2 = K.layers.BatchNormalization()(f_1x1)
        activation2 = K.layers.Activation('relu')(batch_norm2)
        X_next = K.layers.Conv2D(filters=growth_rate,
                                 kernel_size=3,
                                 padding='same',
                                 kernel_initializer=k_init)(activation2)
        concat = K.layers.concatenate([X, X_next])
        X = concat
        nb_filters += growth_rate

    return concat, nb_filters
