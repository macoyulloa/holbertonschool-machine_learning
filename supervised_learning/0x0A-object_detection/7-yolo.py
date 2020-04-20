#!/usr/bin/env python3
"""Yolo algorithm construction"""
import cv2
import os
import glob
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
            grid_h, grid_w, nb_box, _ = outputs[i].shape

            box_conf = 1 / (1 + np.exp(-(outputs[i][:, :, :, 4:5])))
            box_confidence.append(box_conf)
            box_prob = 1 / (1 + np.exp(-(outputs[i][:, :, :, 5:])))
            box_class_probs.append(box_prob)

            box_xy = 1 / (1 + np.exp(-(outputs[i][:, :, :, :2])))
            box_wh = np.exp(outputs[i][:, :, :, 2:4])
            anchors_tensor = self.anchors.reshape(1, 1,
                                                  self.anchors.shape[0],
                                                  nb_box, 2)
            box_wh = box_wh * anchors_tensor[:, :, i, :, :]

            col = np.tile(np.arange(0, grid_w),
                          grid_h).reshape(grid_h, grid_w)
            row = np.tile(np.arange(0, grid_h),
                          grid_w).reshape(grid_w, grid_h).T
            col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
            row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
            grid = np.concatenate((col, row), axis=3)

            box_xy += grid
            box_xy /= (grid_w, grid_h)
            input_h = self.model.input.shape[2].value
            input_w = self.model.input.shape[1].value
            box_wh /= (input_w, input_h)
            box_xy -= (box_wh / 2)
            box_xy1 = box_xy
            box_xy2 = box_xy1 + box_wh
            box = np.concatenate((box_xy1, box_xy2), axis=-1)

            box[..., 0] *= image_size[1]
            box[..., 2] *= image_size[1]
            box[..., 1] *= image_size[0]
            box[..., 3] *= image_size[0]

            boxes.append(box)

        return ((boxes, box_confidence, box_class_probs))

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filtering the boxes with object threshold
        Arg:
            boxes: np shape (grid_height, grid_width, anchor_boxes, 4)
                   with the process outputs boxes
            box_confidences: shape (grid_height, grid_width, anchor_boxes, 1)
                   has all the processed box confidences
            box_class_probs: (grid_height, grid_width, anchor_boxes, classes)
                   has all the processed box class probabilities
        Returns tuple of (filtered_boxes, box_classes, box_scores)
            filtered_boxes: np array shape (?, 4) all filtered bounding boxes
            box_classes: np array shape (?,) class number that each box in
                         filtered_boxes predicts
            box_scores: np array of shape (?) the box scores for each box
                        in filtered_boxes
        """
        box_score = []
        for confis, probs in zip(box_confidences, box_class_probs):
            score = confis * probs
            box_score.append(score)

        box_class_scores = [score.max(axis=-1) for score in box_score]
        box_score_list = [box.reshape(-1) for box in box_class_scores]
        box_scores_conca = np.concatenate(box_score_list, axis=-1)

        filt = np.where(box_scores_conca >= self.class_t)

        box_scores = box_scores_conca[filt]

        boxes = [box.reshape(-1, 4) for box in boxes]
        boxes_conca = np.concatenate(boxes, axis=0)
        filtered_boxes = boxes_conca[filt]

        box_class = [score.argmax(axis=-1) for score in box_score]
        box_class_list = [box.reshape(-1) for box in box_class]
        box_class_conca = np.concatenate(box_class_list, axis=-1)
        box_classes = box_class_conca[filt]

        return (filtered_boxes, box_classes, box_scores)

    def keeped_boxes_iou(self, filtered_boxes, box_scores):
        """ selections of the boxes to keep by its score index """
        x = filtered_boxes[:, 0]
        y = filtered_boxes[:, 1]
        w = filtered_boxes[:, 2]
        h = filtered_boxes[:, 3]

        areas = w * h

        order = box_scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_t)[0]
            order = order[inds + 1]

        keep = np.array(keep)
        return(keep)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ Non-max suppression to take the best box option for obj detection
        Arg:
            filtered_boxes: np shape (?, 4) containing all of the
                            filtered bounding boxes.
            box_classes: np shape (?,) containing the class number
                         that the filtered_boxes predicts
            box_scores: np shape (?) box scores for each box in filtered_boxes
        Returns:(box_predic, predic_box_classes, predic_box_scores)
            box_predictions: np shape (?, 4) all of the predicted
                             bounding boxes ordered by class and box score
            predicted_box_classes: np shape (?,) with class number for
                                   box_predictions
            predicted_box_scores: np shape (?) with the box scores for
                                  box_predictions
        """
        nboxes, nclasses, nscores = [], [], []
        for c in set(box_classes):
            inds = np.where(box_classes == c)
            b = filtered_boxes[inds]
            c = box_classes[inds]
            s = box_scores[inds]

            keep = self.keeped_boxes_iou(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        boxes_predic = np.concatenate(nboxes)
        classes_predic = np.concatenate(nclasses)
        scores_predic = np.concatenate(nscores)

        return (boxes_predic, classes_predic, scores_predic)

    def load_images(self, folder_path):
        """load the images from a folder_path
        Arg:
             folder_path: path to the folder holding all the images to load
        Returns a tuple of (images, image_paths)
             images: a list of images as numpy.ndarray
             image_paths: list paths to the individual images in images
        """
        images = []

        path = folder_path + '/*.jpg'
        image_paths = glob.glob(path, recursive=False)
        for img in image_paths:
            image = cv2.imread(img)
            images.append(image)
        return (images, image_paths)

    def preprocess_images(self, images):
        """resizing the images
        Arg:
            images: a list of images as numpy.ndarrays
            Resize the images with inter-cubic interpolation
            Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
            pimages: np shape (ni, input_h, input_w, 3). Preprocessed images
                    ni: the number of images that were preprocessed
                    input_h: the input height for the Darknet model
                    input_w: the input width for the Darknet model
                    3: number of color channels
            image_shapes: np of shape (ni, 2) has the original height and
                          width of the images
                          2 => (image_height, image_width)
        """
        images_dimensions = [img.shape[:2] for img in images]
        images_shapes = np.stack(images_dimensions, axis=0)

        input_h = self.model.input.shape[1].value
        input_w = self.model.input.shape[2].value

        images_resized = [cv2.resize(img,
                                     (input_h, input_w),
                                     interpolation=cv2.INTER_CUBIC)
                          for img in images]
        images_rescaled = [img/255 for img in images_resized]

        pimages = np.stack(images_rescaled, axis=0)

        return (pimages, images_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """ Displays a image with all boundary boxes, class names,
            and box scores
        Arg:
            - image: np.nndarray with a unprocessed image
            - boxes: a numpy.ndarray the boundary boxes for the image
            - box_classes: a numpy.ndarray class indices for each box
            - box_scores: a numpy.ndarray the box scores for each box
            - file_name: file path where the original image is stored
        Only the s key is pressed:
            - The image should be saved in the directory detections,
            located in the current directory
        """
        box_scores_round = np.around(box_scores, decimals=2)
        for i, box in enumerate(boxes):
            x, y, w, h = box
            text_class = self.class_names[box_classes[i]]
            text_score = str(box_scores_round[i])
            start_rectangle = (int(x), int(h))
            start_text = (int(x) + 1, int(y) - 6)
            end_rectangle = (int(w), int(y))
            cv2.rectangle(image,
                          start_rectangle,
                          end_rectangle,
                          (255, 0, 0), 2)
            cv2.putText(image,
                        text_class + " " + text_score,
                        start_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1,
                        cv2.LINE_AA)
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists("./detections"):
                os.makedirs("./detections")
            cv2.imwrite("./detections/" + file_name, image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """ object detection prediction from images
        Arg:
            folder_path: str with the path to the folder holding all
                     the images to predict
        Returns: a tuple of (predictions, image_paths):
                 predictions: list of tuples for each image of
                              (boxes, box_classes, box_scores)
                 image_paths: list of image paths corresponding
                              to each prediction in predictions
        """
        predictions = []
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        outs = self.model.predict(pimages)

        for i, img in enumerate(images):
            outputs = [outs[0][i, :, :, :, :],
                       outs[1][i, :, :, :, :],
                       outs[2][i, :, :, :, :]]
            img_original_dimensions = np.array([img.shape[0],
                                                img.shape[1]])
            bxs, bx_confcs, bx_class_probs = self.process_outputs(
                outputs, img_original_dimensions)
            boxes, box_classes, box_scores = self.filter_boxes(
                bxs, bx_confcs, bx_class_probs)
            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes, box_classes, box_scores)
            name = image_paths[i].split('/')[-1]
            self.show_boxes(img, boxes, box_classes, box_scores, name)
            predictions.append((boxes, box_classes, box_scores))

        return (predictions, image_paths)
