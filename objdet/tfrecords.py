# Copyright (c) 2019
# Manuel Cherep <manuel.cherep@epfl.ch>

"""
Functions to create TFRecords
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def create_tfrecords(data_path, model_name):
    """ Creates the corresponding tfrecord for the data """

    # New path for gfile in tensorflow
    tf.gfile = tf.io.gfile

    label_map_path = os.path.join(model_name, LABEL_MAP)
    label_map = label_map_util.create_category_index_from_labelmap(label_map_path,
                                                                   use_display_name=True)

    for subdir in os.listdir(data_path):
        files = os.listdir(os.path.join(data_path, subdir))
        tfrecord_path = os.path.join(data_path,
                                     subdir,
                                     'tf_' + subdir + '.record')
        writer = tf.io.TFRecordWriter(tfrecord_path)
        labels_df = pd.read_csv(os.path.join(data_path,
                                             subdir,
                                             'labels.csv'))

        for filename in files:
            if '.jpg' in filename:
                filename_full_path = os.path.join(data_path, subdir, filename)
                tf_example = create_tf_example(filename_full_path,
                                               labels_df,
                                               label_map)
                writer.write(tf_example.SerializeToString())

        writer.close()


def create_tf_example(filename, labels_df, label_map):
    """ Creates a TF Example for each image """

    # Full path to the image
    img = matplotlib.image.imread(filename)
    boxes_df = labels_df[labels_df.frame == filename.split('/')[-1]]

    height = img.shape[1]  # Image height
    width = img.shape[0]  # Image width
    encoded_image_data = img.tobytes()  # Encoded image bytes
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for index, row in boxes_df.iterrows():
        # List of normalized coordinates in bounding box (1 per box)
        xmins.append(row.xmin / width)
        xmaxs.append(row.xmax / width)
        ymins.append(row.ymin / height)
        ymaxs.append(row.ymax / height)
        classes_text.append(label_map.get(row.class_id)
                            ['name'].encode('utf-8'))
        classes.append(row.class_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
