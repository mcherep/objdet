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
import cv2


def create_tfrecords(data_path, classes):
    """ Creates the corresponding tfrecord for the data """

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
                                               classes)
                writer.write(tf_example.SerializeToString())

        writer.close()


def create_tf_example(filename, labels_df, classes):
    """ Creates a TF Example for each image """

    # Full path to the image
    img = matplotlib.image.imread(filename)
    boxes_df = labels_df[labels_df.frame == filename.split('/')[-1]]

    height = img.shape[0]  # Image height
    width = img.shape[1]  # Image width
    image_format = b'jpg'

    with tf.gfile.FastGFile(filename, 'rb') as fid:
        image_data = fid.read()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes_ids = []
    for index, row in boxes_df.iterrows():
        # List of normalized coordinates in bounding box (1 per box)
        xmins.append(row.xmin / width)
        xmaxs.append(row.xmax / width)
        ymins.append(row.ymin / height)
        ymaxs.append(row.ymax / height)
        classes_text.append(classes.get(row.class_id).encode('utf-8'))
        classes_ids.append(row.class_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example
