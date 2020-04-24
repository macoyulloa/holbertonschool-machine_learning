#!/usr/bin/env python3
""" triple loss model """

import tensorflow


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
