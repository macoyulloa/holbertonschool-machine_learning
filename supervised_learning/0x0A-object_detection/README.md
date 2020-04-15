0x0A. Object Detection
======================

Tasks
-----

#### 0\. Initialize Yolo

class `Yolo` that uses the Yolo v3 algorithm to perform object detection:

-   class constructor: `def __init__(self, model_path, classes_path, class_t, nms_t, anchors)`


#### 1\. Process Outputs

class `Yolo` (Based on `0-yolo.py`):

-   Add the public method `def process_outputs(self, outputs, image_size)`
-   Returns a tuple of `(boxes, box_confidences, box_class_probs)`:
        -   `boxes`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 4)` containing the processed boundary boxes for each output, respectively:
            -   `4` => `(x1, y1, x2, y2)`
            -   `(x1, y1, x2, y2)` should represent the boundary box relative to original image
        -   `box_confidences`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 1)` containing the box confidences for each output, respectively
        -   `box_class_probs`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, classes)` containing the box's class probabilities for each output, respectively

*HINT: The Darknet model is an input to the class for a reason. It may not always have the same number of outputs, input sizes, etc.*


#### 2\. Filter Boxes

class `Yolo` (Based on `1-yolo.py`):

-   Add the public method `def filter_boxes(self, boxes, box_confidences, box_class_probs)`


#### 3\. Non-max Suppression

Write a class `Yolo` (Based on `2-yolo.py`)

-   Add the public method `def non_max_suppression(self, filtered_boxes, box_classes, box_scores)`


#### 4\. Load images

class `Yolo` (Based on `3-yolo.py`):

-   Add the static method `def load_images(folder_path):`
    -   `folder_path`: a string representing the path to the folder holding all the images to load
    -   Returns a tuple of `(images, image_paths)`:
        -   `images`: a list of images as `numpy.ndarray`s
        -   `image_paths`: a list of paths to the individual images in `images`


#### 5\. Preprocess images

class `Yolo` (Based on `4-yolo.py`):

-   Add the public method `def preprocess_images(self, images):`
    -   `images`: a list of images as `numpy.ndarray`s
    -   Resize the images with inter-cubic interpolation
    -   Rescale all images to have pixel values in the range `[0, 1]`
    -   Returns a tuple of `(pimages, image_shapes)`:
        -   `pimages`: a `numpy.ndarray` of shape `(ni, input_h, input_w, 3)` containing all of the preprocessed images
            -   `ni`: the number of images that were preprocessed
            -   `input_h`: the input height for the Darknet model *Note: this can vary by model*
            -   `input_w`: the input width for the Darknet model *Note: this can vary by model*
            -   `3`: number of color channels
        -   `image_shapes`: a `numpy.ndarray` of shape `(ni, 2)` containing the original height and width of the images
            -   `2` => `(image_height, image_width)`


#### 6\. Show boxes

class `Yolo` (Based on `5-yolo.py`):

-   Add the public method `def show_boxes(self, image, boxes, box_classes, box_scores, file_name):`
    -   `image`: a `numpy.ndarray` containing an unprocessed image
    -   `boxes`: a `numpy.ndarray` containing the boundary boxes for the image
    -   `box_classes`: a `numpy.ndarray` containing the class indices for each box
    -   `box_scores`: a `numpy.ndarray` containing the box scores for each box
    -   `file_name`: the file path where the original image is stored
    -   Displays the image with all boundary boxes, class names, and box scores *(see example below)*
        -   Boxes should be drawn as with a blue line of thickness 2
        -   Class names and box scores should be drawn above each box in red
            -   Box scores should be rounded to 2 decimal places
            -   Text should be written 5 pixels above the top left corner of the box
            -   Text should be written in `FONT_HERSHEY_SIMPLEX`
            -   Font scale should be 0.5
            -   Line thickness should be 1
            -   You should use `LINE_AA` as the line type
        -   The window name should be the same as `file_name`
        -   If the `s` key is pressed:
            -   The image should be saved in the directory `detections`, located in the current directory
            -   If `detections` does not exist, create it
            -   The saved image should have the file name `file_name`
            -   The image window should be closed
        -   If any key besides `s` is pressed, the image window should be closed without saving


#### 7\. Predict

Write a class `Yolo` (Based on `6-yolo.py`):

-   Add the public method `def predict(self, folder_path):`
    -   `folder_path`: a string representing the path to the folder holding all the images to predict
    -   All image windows should be named after the corresponding image filename without its full path*(see examples below)*
    -   Displays all images using the `show_boxes` method
    -   Returns: a tuple of `(predictions, image_paths)`:
        -   `predictions`: a list of tuples for each image of `(boxes, box_classes, box_scores)`
        -   `image_paths`: a list of image paths corresponding to each prediction in `predictions`
