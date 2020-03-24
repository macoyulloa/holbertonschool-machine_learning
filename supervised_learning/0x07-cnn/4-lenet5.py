#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import tensorflow as tf


def lenet5(x, y):
    """LeNet-5 architecture using tf. 3D image, RGB image - color
    Arg:
       x: tf.placeholder of shape (m, 28, 28, 1)
          containing the input images
       y: f.placeholder of shape (m, 10)
          containing the one-hot labels
    Return:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    k_init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    layer_1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                               padding='same',
                               activation=activation,
                               kernel_initializer=k_init)(x)

    pool_1 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                    strides=2)(layer_1)

    layer_2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                               padding='valid',
                               activation=activation,
                               kernel_initializer=k_init)(pool_1)

    pool_2 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                    strides=2)(layer_2)

    flatten = tf.layers.Flatten()(pool_2)

    layer_3 = tf.layers.Dense(units=120, activation=activation,
                              kernel_initializer=k_init)(flatten)

    layer_4 = tf.layers.Dense(units=84, activation=activation,
                              kernel_initializer=k_init)(layer_3)

    output_layer = tf.layers.Dense(units=10, activation=activation,
                                   kernel_initializer=k_init)(layer_4)

    y_pred = tf.nn.softmax(output_layer)

    loss = tf.losses.softmax_cross_entropy(
        y, output_layer, reduction=tf.losses.Reduction.MEAN)

    train = tf.train.AdamOptimizer().minimize(loss)

    equality = tf.equal(tf.argmax(y, axis=1),
                        tf.argmax(output_layer, axis=1))
    acc = tf.reduce_mean(tf.cast(equality, tf.float32))

    return y_pred, train, loss, acc
