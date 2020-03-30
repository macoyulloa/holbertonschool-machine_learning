#!/usr/bin/env python3
"""Deep Convolutional Architectures"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """identity block
    Arg:
       A_prev: output from the previus layer
       filters: tuple or list containing the filter - F11, F3, F12
          -  F11 is the number of filters in the first 1x1 convolution
          -  F3 is the number of filters in the 3x3 convolution
          -  F12 is the number of filters in the second 1x1 convolution
    Returns: activate output of the identity block
    """
    F11, F3, F12 = filters
    activation = 'relu'
    k_init = K.initializers.he_normal(seed=None)

    l1 = K.layers.Conv2D(filters=F11, kernel_size=1,
                         padding='same',
                         kernel_initializer=k_init)(A_prev)

    batch_norm1 = K.layers.BatchNormalization()(l1)

    activation1 = K.layers.Activation('relu')(batch_norm1)

    l2 = K.layers.Conv2D(filters=F3, kernel_size=3,
                         padding='same',
                         kernel_initializer=k_init)(activation1)

    batch_norm2 = K.layers.BatchNormalization()(l2)

    activation2 = K.layers.Activation('relu')(batch_norm2)

    l3 = K.layers.Conv2D(filters=F12, kernel_size=1,
                         padding='same',
                         kernel_initializer=k_init)(activation2)

    batch_norm3 = K.layers.BatchNormalization()(l3)

    add = K.layers.Add()([batch_norm3, A_prev])

    activation3 = K.layers.Activation('relu')(add)
    return activation3
