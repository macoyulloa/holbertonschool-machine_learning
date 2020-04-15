#!/usr/bin/env python3
"""Yolo algorithm construction"""

import tensorflow.keras as K
import numpy as np


class Yolo():
    """class Yolo that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ initialization of the variables
        Arg:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path with the list of class names used,
                          listed in order of index, can be found
            class_t: float, box score threshold for the init filtering step
            nms_t: float representing the IOU threshold for non-max suppression
            anchors: np.ndarray.shape (outputs, anchor_boxes, 2)
                     has all anchor boxes:
                     outputs: number of outputs (predictions)
                     anchor_boxes: num of anchor boxes used for each prediction
                     2 => [anchor_box_width, anchor_box_height]
        """
        class_names = []
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as classes_file:
            class_names = classes_file.readlines()
        class_names = [x.strip() for x in class_names]

        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ process the outputs
        Arg:
            outputs: list of numpy.ndarrays containing the predictions:
                Each output will have the shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
                     grid_height & grid_width =>
                     the height and width of the grid used for the output
                     anchor_boxes => the number of anchor boxes used
                     4 => (t_x, t_y, t_w, t_h)
                     1 => box_confidence
                     classes => class probabilities for all classes
            image_size: numpy.ndarray image’s original size [height, width]
        Returns: tuple of (boxes, box_confidences, box_class_probs):
                 boxes: np shape (grid_height, grid_width, anchor_boxes, 4)
                 box_confidences: np.shape (grid_height, grid_width,
                    anchor_boxes, 1) has box confidences for each output
                 box_class_probs:(grid_height, grid_width, anchor_boxes,
                    classes) box’s class probabilities for each output
        """
        boxes = []
        box_confidence = []
        box_class_probs = []

        for i in range(len(outputs)):
            box_conf = 1 / (1 + np.exp(-(outputs[i][:, :, :, 4:5])))
            box_confidence.append(box_conf)
            box_prob = 1 / (1 + np.exp(-(outputs[i][:, :, :, 5:])))
            box_class_probs.append(box_prob)

        boxes = [out[..., :4] for out in outputs]
        for i, box in enumerate(boxes):
            grid_h, grid_w, n_anchors, _ = box.shape

            m_h = np.arange(grid_h).reshape(1, grid_h)
            m_h = np.repeat(m_h, grid_w, axis=0).T
            m_h = np.repeat(m_h[:, :, np.newaxis], n_anchors, axis=2)
            m_w = np.arange(grid_w).reshape(1, grid_w)
            m_w = np.repeat(m_w, grid_h, axis=0)
            m_w = np.repeat(m_w[:, :, np.newaxis], n_anchors, axis=2)

            box[..., :2] = 1 / (1 + np.exp(-(outputs[i][:, :, :, :2])))
            box[..., 0] += m_w
            box[..., 1] += m_h

            box[..., 2:] = np.exp(box[..., 2:])
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            box[..., 2] *= anchor_w
            box[..., 3] *= anchor_h

            box[..., 0] /= grid_w
            box[..., 1] /= grid_h

            box[..., 2] /= self.model.input.shape[1].value
            box[..., 3] /= self.model.input.shape[2].value

            box[..., 0] -= box[..., 2] / 2
            box[..., 1] -= box[..., 3] / 2

            box[..., 2] += box[..., 0]
            box[..., 3] += box[..., 1]

            box[..., 0] *= image_size[1]
            box[..., 2] *= image_size[1]
            box[..., 1] *= image_size[0]
            box[..., 3] *= image_size[0]

        return ((boxes, box_confidence, box_class_probs))
