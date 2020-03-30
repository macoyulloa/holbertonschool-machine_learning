#!/usr/bin/env python3
"""Deep Convolutional Architectures"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """transition layer - DenseNet block
    Arg:
       -  X: is the output from the previous layer
       -  nb_filters: int the number of filters in X
       -  compression compression factor transition layer
    Returns: output of the transition layer and the n filters
    """
    k_init = K.initializers.he_normal(seed=None)
    n_f = int(nb_filters * compression)

    batch_norm1 = K.layers.BatchNormalization()(X)
    activation1 = K.layers.Activation('relu')(batch_norm1)
    l_1x1 = K.layers.Conv2D(filters=n_f,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=k_init)(activation1)
    l_avg_pool = K.layers.AveragePooling2D(pool_size=[2, 2],
                                           strides=2,
                                           padding='same')(l_1x1)
    return l_avg_pool, n_f
