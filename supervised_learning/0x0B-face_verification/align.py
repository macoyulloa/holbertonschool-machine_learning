#!/usr/bin/env python3
""" face verification system """

import dlib


class FaceAlign():
    def __init__(self, shape_predictor_path):
        """ initializing the variables of the model
        Arg:
            shape_predictor_path: path to the dlib shape predictor model
        Sets the public instances attributes:
            detector - contains dlib‘s default face detector
            shape_predictor - contains the dlib.shape_predictor
        """
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(shape_predictor_path)

        self.detector = detector
        self.shape_predictor = shape_predictor
