#!/usr/bin/env python3
"""Generative Adversarial Networks"""

import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_generator(Z):
    """ creates the loss tensor and training op for the generator:

    Arg:
    - X: is the tf.placeholder that is the real input for the discriminator

    Returns: loss, train_op
        - loss: is the generator loss
        - train_op: is the training operation for the generator
    """
    G_fake = generator(Z)

    D_G_fake = discriminator(G_fake)

    Generator_loss = - tf.reduce_mean(tf.log(D_G_fake))

    generator_vars = [var for var in tf.trainable_variables()
                      if var.name.startswith("gene")]

    D_train_op = tf.train.AdamOptimizer().minimize(
        Generator_loss, var_list=generator_vars)

    return (Generator_loss, D_train_op)
