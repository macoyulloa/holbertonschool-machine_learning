#!/usr/bin/env python3
""" triple loss model """

import tensorflow
import tensorflow as tf
import numpy as np


class TripletLoss(tensorflow.keras.layers.Layer):
    """custom layer class triplet loss that inherits from
    tensorflow.keras.layers.Layer
    """
    def __init__(self, alpha, **kwargs):
        """ initialization of the variables
        Arg:
            - alpha: is the alpha value used to calculate the triplet loss
        """
        self.alpha = alpha
        super(TripletLoss, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        """ triplet loss values
        Arg:
            - inputs: list with the anchor, positive and negative output
                      tensors from the last layer of the model.
        Return: a tensor containing the triplet loss values
        """
        anchor_output, positive_output, negative_output = inputs
        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

        min_value = tf.convert_to_tensor(0.0, dtype=tf.float64)
        loss = tf.maximum(min_value, self.alpha + d_pos - d_neg)

        return loss

    def call(self, inputs):
        """ call function from my own keras layer
        Arg:
            inputs: list containing the anchor, positive, and negative
                    output tensors from the last layer of the model
        return: the triplet loss tensor
        """
        return K.(inputs, self.loss)
