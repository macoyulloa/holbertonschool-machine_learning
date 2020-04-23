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
        try:
            faces = self.detector(image, 1)
            maxArea = 0

            if len(faces) == 0:
                # non faces was detected
                rectangle_face = dlib.rectangle(left=0,
                                                top=0,
                                                right=image.shape[1],
                                                bottom=image.shape[0])

            if len(faces) >= 1:
                # if one face was detected: rectangle_face = faces[0]
                # multiple faces was detected, take the max box
                for (i, rect) in enumerate(faces):
                    if rect.area() > maxArea:
                        maxArea = rect.area()
                        rectangle_face = rect

            return rectangle_face
        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """ finding the facial landmark, based int the 68 face point
            convert the landmark dlib coord into np.ndarray (x, y) coord
        Arg:
            image: np.ndarray from which to find facial landmarks
            detection: dlib.rectangle with the boundary box of the face
        Returns: np.ndarray shape (p, 2) containing the landmark points
                 None in failure
                 p: number of landmark points
                 2: is the x and y coordinates of the point
        """
        landmark_coord = self.shape_predictor(image, detection)
        if not landmark_coord:
            return None

        coords = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            coords[i] = [landmark_coord.part(i).x,
                         landmark_coord.part(i).y]

        return coords

    def align(self, image, landmark_indices, anchor_points, size=96):
        """ aligns an image for face verification
        Arg:
            image: np.ndarray containing the image to be aligned
            landmark_indices: np.ndarray of shape (3,) with indices of the
                   three landmark points for the affine transformation
            anchor_points: np.ndarray of shape (3, 2) with destination
                   points for affine transform, scaled to the range [0, 1]
            size: is the desired size of the aligned image
        Returns: np.ndarray shape (size, size, 3) with the aligned image
                 or None if no face is detected
        """
        box = self.detect(image)
        landmarks = self.find_landmarks(image, box)

        pts_eyeL_eyeR_nose = landmarks[landmark_indices]
        pts_eyeL_eyeR_nose = pts_eyeL_eyeR_nose.astype('float32')
        anchor_scaled = anchor_points * size
        warp_mat = cv2.getAffineTransform(pts_eyeL_eyeR_nose, anchor_scaled)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

        return warp_dst
