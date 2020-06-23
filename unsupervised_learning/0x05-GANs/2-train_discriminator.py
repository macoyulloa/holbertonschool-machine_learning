#!/usr/bin/env python3
"""Generative Adversarial Networks"""

import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_discriminator(Z, X):
    """ creates the loss tensor and training op for the discriminator:

    Arg:
    - Z: is the tf.placeholder that is the input for the generator
    - X: is the tf.placeholder that is the real input for the discriminator

    Returns: loss, train_op
        - loss: is the discriminator loss
        - train_op: is the training operation for the discriminator
    """
    D_fake = generator(Z)

    D_real = discriminator(X)
    D_G_fake = discriminator(D_fake)

    D_loss = - tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_G_fake))

    discri_vars = [var for var in tf.trainable_variables()
                   if var.name.startswith("disc")]

    D_train_op = tf.train.AdamOptimizer().minimize(
        D_loss, var_list=discri_vars)

    return (D_loss, D_train_op)
