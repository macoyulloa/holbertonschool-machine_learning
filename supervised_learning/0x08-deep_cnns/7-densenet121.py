#!/usr/bin/env python3
"""Deep Convolutional Architectures"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """DenseNet121 architecture
    Arg:
       - growth_rate: number of channels of each layer in the dense block
       - compression: model compactness intransition layer
    Functions:
       - dense block
       - transition block
    Returns: Keras model
    """
    activation = 'relu'
    k_init = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))
    batch_norm0 = K.layers.BatchNormalization()(X)
    activation0 = K.layers.Activation('relu')(batch_norm0)

    l1 = K.layers.Conv2D(filters=64, kernel_size=7,
                         strides=2,
                         padding='same',
                         kernel_initializer=k_init)(activation0)

    l_pool1 = K.layers.MaxPooling2D(pool_size=[3, 3],
                                    strides=2,
                                    padding='same')(l1)

    l2, n_filt_2 = dense_block(l_pool1, 64, growth_rate, 6)

    l3, n_filt_3 = transition_layer(l2, n_filt_2, compression)

    l4, n_filt_4 = dense_block(l3,  n_filt_3, growth_rate, 12)

    l5, n_filt_5 = transition_layer(l4,  n_filt_4, compression)

    l6, n_filt_6 = dense_block(l5,  n_filt_5, growth_rate, 24)

    l7, n_filt_7 = transition_layer(l6,  n_filt_6, compression)

    l8, n_filt_8 = dense_block(l7,  n_filt_7, growth_rate, 16)

    l_avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7],
                                           strides=7,
                                           padding='same')(l8)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=k_init)(l_avg_pool)

    model = K.models.Model(inputs=X, outputs=Y)
    return model
