#!/usr/bin/env python3
"""Deep Convolutional Architectures"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ResNet50 architecture
    Functions:
       - identity block
       - projection block
    Returns: Keras model
    """
    activation = 'relu'
    k_init = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    l1 = K.layers.Conv2D(filters=64, kernel_size=7,
                         strides=2,
                         padding='same',
                         kernel_initializer=k_init)(X)

    batch_norm1 = K.layers.BatchNormalization()(l1)

    activation1 = K.layers.Activation('relu')(batch_norm1)

    l_pool1 = K.layers.MaxPooling2D(pool_size=[3, 3],
                                    strides=2,
                                    padding='same')(activation1)

    l2 = projection_block(l_pool1, [64, 64, 256], 1)

    l3 = identity_block(l2, [64, 64, 256])

    l4 = identity_block(l3, [64, 64, 256])

    l5 = projection_block(l4, [128, 128, 512])

    l6 = identity_block(l5, [128, 128, 512])

    l7 = identity_block(l6, [128, 128, 512])

    l8 = identity_block(l7, [128, 128, 512])

    l9 = projection_block(l8, [256, 256, 1024])

    l10 = identity_block(l9, [256, 256, 1024])

    l11 = identity_block(l10, [256, 256, 1024])

    l12 = identity_block(l11, [256, 256, 1024])

    l13 = identity_block(l12, [256, 256, 1024])

    l14 = identity_block(l13, [256, 256, 1024])

    l15 = projection_block(l14, [512, 512, 2048])

    l16 = identity_block(l15, [512, 512, 2048])

    l17 = identity_block(l16, [512, 512, 2048])

    l_avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7],
                                           strides=7,
                                           padding='same')(l17)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=k_init)(l_avg_pool)
    model = K.models.Model(inputs=X, outputs=Y)
    return model
