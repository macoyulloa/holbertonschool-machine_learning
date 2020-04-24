#!/usr/bin/env python3
""" trains the model triple loss model """

from triplet_loss import TripletLoss
import tensorflow as tf


class TrainModel():
    """trains the model for face verification using triplet loss
    """
    def __init__(self, model_path, alpha):
        """ initialization of the variables
        Arg:
            - model_path: parth to the base face verification embedding mode
                 - loads model: tf.keras.utils.CustomObjectScope({'tf': tf}):
                 - saves this model as the public instance method base_model
            - alpha: is the alpha value used to calculate the triplet loss
        Creates a new model:
            inputs: [A, P, N]
                 numpy.ndarrys with the anchor, positive and negatives images
            outputs: the triplet losses of base_model
            compiles the model with Adam optimization and no losses
            save this model as the public instance method training_model
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = tf.keras.models.load_model(model_path)

        self.alpha = alpha

        tl = TripletLoss(0.2)
        output = tl(inputs)
        training_model = tf.keras.models.Model([A, P, N], output)

        training_model.compile(optimizer='Adam')

        training_model.save()
