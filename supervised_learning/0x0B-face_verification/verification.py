#!/usr/bin/env python3
"""Yolo algorithm construction"""

import tensorflow.keras as K
import numpy as np


class FaceVerification():
    """Face Verification class
    """
    def __init__(self, model, database, identities):
        """initialized the face verification variables
        Arg:
        model: is the face verification embedding model or the path to
               where the model is stored
        database: numpy.ndarray of all the face embeddings in the database
        identities: list corresponding to the embeddings in the database
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = tf.keras.models.load_model(model)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        """ get the embeddings
        Arg:
            - images are the images to retrieve the embeddings of
        Returns: a numpy.ndarray of embeddings
        """
        return embeddings

    def verify(self, image, tau=0.5):
        """ verify method for face recognition
        Arg:
        - image is the aligned image of the face to be verify
        - tau is the maximum euclidean distance used for verification
        Returns: (identity, distance), or (None, None) on failure
        """
        return identity, distance
