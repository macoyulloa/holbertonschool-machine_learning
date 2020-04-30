#!/usr/bin/env python3
"""Neural Style Tranfer"""

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
        self.model = self.load_model()

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

    def load_model(self):
        """ load the model used to calculate the cost
            saved the model in the instance attribute model
        """
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                pooling='avg',
                                                weights='imagenet')

        style_outputs, content_outputs = [], []
        # Get output layers corresponding to style and content layers
        for layer in vgg.layers:
            layer.trainable = False
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name in self.content_layer:
                content_outputs.append(layer.output)

        model_outputs = style_outputs + content_outputs
        # Build model
        model = tf.keras.models.Model(vgg.input, model_outputs)

        return model

    @staticmethod
    def gram_matrix(input_layer):
        """ calculate the gram matrices
        Arg:
         - input_layer: instance of tf.Tensor or tf.Variable of shape
                    (1, h, w, c) with layer.output to calculate gram matrix
        Returns: tf.Tensor shape (1, c, c) with gram matrix of input_layer
        """
        if not isinstance(input_layer, tf.Tensor) or isinstance(
                input_layer, tf.Variable):
            if tf.rank(input_layer) is not tf.constant(4, tf.int32):
                raise TypeError(
                    "input_layer must be a tensor of rank 4")

        channels = int(input_layer.shape[-1])
        activation = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(activation)[0]
        gram = tf.matmul(activation, activation, transpose_a=True)
        gram_matrix = gram / tf.cast(n, tf.float32)
        gram_matrix = tf.expand_dims(gram_matrix, axis=0)

        return gram_matrix
