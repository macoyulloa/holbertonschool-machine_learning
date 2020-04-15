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
        model = K.models.load_model(model_path)
        with open(classes_path, "r") as classes_file:
            class_names = classes_file.readlines()
        class_names = [x.strip() for x in class_names]

        self.model = model
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
        img_h = image_size[0]
        img_w = image_size[1]
        boxes = []
        box_confidence = []
        box_class_probs = []

        for i in range(len(outputs)):
            grid_h, grid_w, nb_box, coor_pc_classes = outputs[i].shape

            box_conf = 1 / (1 + np.exp(-(outputs[i][:,:,:,4:5])))
            box_confidence.append(box_conf)
            box_prob = 1 / (1 + np.exp(-(outputs[i][:,:,:,5:])))
            box_class_probs.append(box_prob)

            box_xy = 1 / (1 + np.exp(-(outputs[i][:,:,:,:2])))
            box_wh = np.exp(outputs[i][:,:,:,2:4])
            anchors_tensor = self.anchors.reshape(1, 1,
                                                  self.anchors.shape[0],
                                                  nb_box, 2)
            box_wh = box_wh * anchors_tensor[:,:,i,:,:]

            for j in range(grid_h*grid_w):
                row = j / grid_w
                col = j % grid_w

                x = box_xy[...,0]
                y = box_xy[...,1]
                w = box_wh[...,0]
                h = box_wh[...,1]
                x = (col + x) / grid_w
                y = (row + y) / grid_h
                w = w / img_w
                h = h / img_h
            x = (x-w/2)
            y = (y-h/2)
            w = (x+w/2)
            h = (y-h/2)
            print(x.shape)
            print(y.shape)
            print(w.shape)
            print(h.shape)
            box = x.reshape(grid_h, grid_w, 3, 1)
            box = np.concatenate((x, y, w, h), axis=-1)

            #for m in range(anchors_tensor.shape[2]):
            #box_wh = box_wh * anchors_tensor[:,:,m,:,:]

            #print(grid_w)
            #print(grid_h)
            #col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
            #print(col.shape)
            #row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
            #print(row.shape)
            #col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
            #print(col.shape)
            #row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
            #print(row.shape)
            #grid = np.concatenate((col, row), axis=-1)
            #print(grid.shape)

            #box_xy += grid
            #box_xy /= (grid_w, grid_h)
            #box_wh /= (img_w, img_h)
            #box_xy -= (box_wh / 2.)
            #box = np.concatenate((box_xy, box_wh), axis=-1)

            #box = outputs[i][:,:,:,:4]
            boxes.append(box)

        return ((boxes, box_confidence, box_class_probs))
