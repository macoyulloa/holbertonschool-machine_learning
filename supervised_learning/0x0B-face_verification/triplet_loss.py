#!/usr/bin/env python3
""" triple loss model """

import tensorflow
import tensorflow as tf
import tensorflow.keras as K


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
        anchor, positive, negative = inputs
        p_dist = K.backend.sum(K.backend.square(anchor-positive), axis=-1)
        n_dist = K.backend.sum(K.backend.square(anchor-negative), axis=-1)

        return (K.backend.maximum(p_dist - n_dist + self.alpha, 0))

    def call(self, inputs):
        """ call function from my own keras layer
        Arg:
            inputs: list containing the anchor, positive, and negative
                    output tensors from the last layer of the model
        return: the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
