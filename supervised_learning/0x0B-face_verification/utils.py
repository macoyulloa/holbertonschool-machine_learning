#!/usr/bin/env python3
""" Face Varification System """

import numpy as np
import os
import cv2

def load_images(images_path, as_array=True):
    """ Loading the images in a RGB format from a directory or file by
        alphabetical order  by filename.
    Arg:
        images_path: path to a directory from which to load images
        as_array: boolean indicating the images should be
                  loaded as one numpy.ndarray or not
               If True, load the imgs numpy.ndarray of shape (m, h, w, c)
               If False, load the imgs as a list of individual np.ndarrays
    Returns: images, filenames
        images: is either a list/numpy.ndarray of all images
        filenames: is a list of the filenames of each image
    """
    
