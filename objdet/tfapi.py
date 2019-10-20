# Copyright (c) 2019
# Manuel Cherep <manuel.cherep@epfl.ch>

"""
Functions dealing with the Tensorflow Object Detection API
"""

import subprocess
import os
import re
import numpy as np
import urllib.request
import tarfile
import tensorflow as tf
from PIL import Image
from IPython.display import display
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

LABEL_MAP = 'label_map.pbtxt'


def install():
    """ Install all the requirements for the API """

    bashCmd = 'git clone --quiet https://github.com/tensorflow/models.git'
    subprocess.call(bashCmd.split())

    bashCmd = 'pip install Cython contextlib2 pillow lxml matplotlib pycocotools'
    subprocess.call(bashCmd.split())

    bashCmd = 'pip install .'
    subprocess.call(bashCmd.split(), cwd='models/research/')

    # The wildcard forces to use the full command without split.
    # Otherwise, it doesn't work due to escaping
    bashCmd = 'protoc object_detection/protos/*.proto --python_out=.'
    subprocess.run(bashCmd, cwd='models/research/', shell=True, check=True)

    pwd = os.path.join(os.getcwd(), 'models', 'research')
    if 'PYTHONPATH' in os.environ:
        python_path = os.environ['PYTHONPATH']
    else:
        python_path = ""
    os.environ['PYTHONPATH'] = "{}:{}:{}/slim".format(python_path, pwd, pwd)

    bashCmd = 'export PYTHONPATH=:{}:{}/slim'.format(pwd, pwd)
    subprocess.call(bashCmd, shell=True)

    # Get output of command and verify that everything is correct
    bashCmd = 'python object_detection/builders/model_builder_test.py'
    subprocess.run(bashCmd.split(), cwd='models/research', check=True)


def download_model(model_name):
    """ Downloads a model from the zoo """

    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'

    # Download model
    urllib.request.urlretrieve(base_url + model_file, model_file)

    # Untar and clean
    tar = tarfile.open(model_file)
    tar.extractall()
    tar.close()
    os.remove(model_file)


def transfer_learning(classes, model_name, data_path):
    """ Prepares the model config for training in a custom
    dataset for our particular problem """

    # Create label map
    pwd = os.getcwd()
    label_map_path = os.path.join(pwd, model_name, LABEL_MAP)
    f = open(label_map_path, 'w+')
    for idx, class_ in enumerate(classes):
        item = "item {\n id: " + str(idx+1) + "\n name: '" + class_ + "'\n}"
        f.write(item)
        f.write("\n")

    model_ckpt = os.path.join(pwd, model_name, 'model.ckpt')
    train_record = os.path.join(os.path.dirname(pwd),
                                data_path,
                                'train',
                                'tf_train.record')
    val_record = os.path.join(os.path.dirname(pwd),
                              data_path,
                              'validation',
                              'tf_validation.record')
    num_classes = 'num_classes: {}'.format(len(classes))

    # Edit config file for transfer learning
    config = os.path.join(model_name, 'pipeline.config')
    with open(config) as f:
        s = f.read()
    with open(config, 'w') as f:
        s = re.sub('num_classes: [0-9]*',
                   num_classes, s)
        s = re.sub('PATH_TO_BE_CONFIGURED/model.ckpt',
                   model_ckpt, s)
        s = re.sub('PATH_TO_BE_CONFIGURED/mscoco_train.record',
                   train_record, s)
        s = re.sub('PATH_TO_BE_CONFIGURED/mscoco_val.record',
                   val_record, s)
        s = re.sub('PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt',
                   label_map_path, s)
        # This parameter is unnecessary but fails
        # in ssd_mobilenet_v2_coco -> comment
        s = re.sub('batch_norm_trainable: true',
                   '# batch_norm_trainable: true', s)
        f.write(s)


def train(model_path, train_steps, eval_steps):
    """ Trains the given model """
    pipeline_config = os.path.join(model_path, 'pipeline.config')
    model_dir = os.path.join(model_path, 'trained')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    bashCmd = ('python models/research/object_detection/model_main.py'
               ' --pipeline_config_path={}'
               ' --model_dir={}'
               ' --alsologtostderr'
               ' --num_train_steps={}'
               ' --num_eval_steps={}').format(pipeline_config, model_dir, train_steps, eval_steps)
    subprocess.run(bashCmd.split(), check=True)


def save_model(model_path):
    """ Save the given model to be loaded later for inference """

    output_dir = os.path.join(model_path, 'trained')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    lst = os.listdir(output_dir)
    lf = filter(lambda k: 'model.ckpt-' in k, lst)
    last_model = sorted(lf)[-1].replace('.meta', '')

    pipeline_config = os.path.join(model_path, 'pipeline.config')
    trained_checkpoint = os.path.join(output_dir, last_model)
    bashCmd = ('python models/research/object_detection/export_inference_graph.py'
               ' --input_type=image_tensor'
               ' --pipeline_config_path={}'
               ' --trained_checkpoint_prefix={}'
               ' --output_directory={}').format(pipeline_config, trained_checkpoint, output_dir)
    subprocess.run(bashCmd.split(), check=True)


def load_model(model_dir):
    """ Loads a model """

    model_dir = os.path.join(model_dir, 'trained', 'saved_model')
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model


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
