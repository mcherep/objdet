# Copyright (c) 2019
# Manuel Cherep <manuel.cherep@epfl.ch>

"""
Functions to run predictions on models
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from IPython.display import display
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def predict(model, image):
    """ Inference given the model and the image(s) """

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    return output_dict


def plot_prediction(model, img, model_name):
    """ Plots the inference given the model and the image(s) """
    # Patch the location of gfile
    tf.gfile = tf.io.gfile
    # Prediction
    output_dict = predict(model, img)
    label_map_path = os.path.join(model_name, LABEL_MAP)
    label_map = label_map_util.create_category_index_from_labelmap(label_map_path,
                                                                   use_display_name=True)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        label_map,
        use_normalized_coordinates=True,
        line_thickness=8)

    display(Image.fromarray(img))
