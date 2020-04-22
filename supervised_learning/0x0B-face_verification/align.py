#!/usr/bin/env python3
"""face detection/verification system"""

import dlib
import numpy as np
import cv2


class FaceAlign():
    """class FaceAlign to permorfs a face verifications system
    """

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

    def detect(self, image):
        """ detects a face in an image
        Arg:
            image: np.ndarray containing an image from which to detect a face
        Returns: dlib.rectangle containing the boundary box for the face
                 in the image, or None on failure
               - If multiple faces are detected, return the largest area rec
               - If no faces, return dlib.rect that is the same as the image
        """
        faces = self.detector(image, 1)
        if not faces:
            return None
        maxArea = 0

        if len(faces) == 0:
            # non faces was detected
            rectanlge_face = dlib.rectangle(left=0,
                                            top=0,
                                            right=image.shape[1],
                                            bottom=image.shape[0])
        if len(faces) >= 1:
            # if one face was detected: rectangle_face = faces[0]
            # multiple faces was detected, take the max box
            for (i, rect) in enumerate (faces):
                if rect.area() > maxArea:
                    maxArea = rect.area()
                    rectangle_face = rect

        return rectangle_face
