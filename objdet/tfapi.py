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


def transfer_learning(train_steps, classes, model_name, data_path):
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
        # Set the number of steps
        s = re.sub('num_steps: [0-9]*',
                   'num_steps: {}'.format(train_steps), s)
        f.write(s)


def train(model_path, train_dir):
    """ Trains the given model """
    pipeline_config = os.path.join(model_path, 'pipeline.config')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    bashCmd = ('python models/research/object_detection/legacy/train.py'
               ' --pipeline_config_path={}'
               ' --train_dir={}'
               ' --logtostderr').format(pipeline_config, train_dir)
    process = subprocess.run(bashCmd.split(),
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    print(process.stderr)


<  # !python models/research/object_detection/model_main.py \
#     --pipeline_config_path=ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \
#     --model_dir=checkpoints \
#     --num_train_steps=3000 \
#     --sample_1_of_n_eval_examples=1 \
#     --alsologtostderr

# !cp ssd_mobilenet_v2_coco_2018_03_29/pipeline.config checkpoints


def save_model(train_dir):
    """ Save the given model to be loaded later for inference """

    lst = os.listdir(train_dir)
    lf = filter(lambda k: 'model.ckpt-' in k, lst)
    last_model = sorted(lf)[-1].replace('.meta', '')

    pipeline_config = os.path.join(train_dir, 'pipeline.config')
    trained_checkpoint = os.path.join(train_dir, last_model)
    bashCmd = ('python models/research/object_detection/export_inference_graph.py'
               ' --input_type=image_tensor'
               ' --pipeline_config_path={}'
               ' --trained_checkpoint_prefix={}'
               ' --output_directory={}').format(pipeline_config, trained_checkpoint, train_dir)
    subprocess.run(bashCmd.split(), check=True)


def load_model(model_dir):
    """ Loads a model """

    model = tf.saved_model.load_v2(model_dir)
    model = model.signatures['serving_default']
    return model
