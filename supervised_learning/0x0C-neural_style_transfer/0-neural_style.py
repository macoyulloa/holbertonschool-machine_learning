#!/usr/bin/env python3
""" """
import numpy as np
import tensorflow as tf


class NST:
    """performs tasks for neural style transfer
    Class atributes:
        - Content layer where will pull our feature maps
        - Style layer we are interested in
    """

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ initializing the varibles
        Arg:
            - style_image: img used as a style reference, numpy.ndarray
            - content_image: image used as a content reference, numpy.ndarray
            - alpha: the weight for content cost
            - beta: the weight for style cost
        Enviornment:
            Eager execution: TensorFlow’s imperative programming
                             environment, evaluates operations immediately
        """
        if not isinstance(style_image, np.ndarray):
            if len(style_image) != 3 or style_image.shape[2] != 3:
                raise TypeError(
                    "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray):
            if len(content_image) != 3 or content_image.shape[2] != 3:
                raise TypeError(
                    "content_image must be a numpy.ndarray with shape (h, w, 3)"
                )

        if alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()
        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """ rescales an image such that its pixels values are between 0
            and 1 and its largest side is 512 pixels
        Arg:
           - image: np.ndarray (h, w, 3) containing the image to be scaled
        Returns:
           - A scaled image Tensor
        """
        if not isinstance(image, np.ndarray):
            if len(image) != 3 or image.shape[2] != 3:
                raise TypeError(
                    "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape

        if h > w:
            new_h = 512
            new_w = (w / h) * 512
        if w > h:
            new_w = 512
            new_h = (h / w) * 512
        else:
            new_h = 512
            new_w = 512

        image = np.expand_dims(image, axis=0)
        bicubic = tf.image.ResizeMethod.BICUBIC
        img_resiz = tf.image.resize_images(image,
                                           (new_h, new_w),
                                           method=bicubic,
                                           preserve_aspect_ratio=True)
        img_rescaled = tf.div(
            tf.subtract(img_resiz, tf.reduce_min(img_resiz)),
            tf.subtract(tf.reduce_max(img_resiz), tf.reduce_min(img_resiz)))

        return img_rescaled
