# Object Detection in Tensorflow

A collection of functions that allow to perform object detection using any model from the zoo in Tensorflow. The goal is to be able to integrate the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) easily in other projects, allowing transfer learning and training in Google Colab. In order to use it you should choose a model from the [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), and preprocess your data to follow the given format below.

**This project is still under heavy testing. It should be production ready soon**

## Rant

I don't like Tensorflow because I consider it unnecessarily difficult to use. All versions break backwards compatibility, I could print all the future warnings I get and cover myself when I go to sleep, and the word `lightweight` hasn't reached them (to mention a few). I love `Keras`, but in this case it felt like an overkill when most of the things I want to do are already in the Tensorflow Object Detection API. Therefore, I decided to isolate the evil in this repo for others to use it easily. If you've found this repository in desperation, please feel free to share your rant with me. Yours truly. P.D. This code will most likely break in a few months when the developers decide to do some random changes.

## Installation

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Most of the [installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) for the Tensorflow Object Detection API is done in `objdet/tfapi.py`. However, you still need to install `protobuf` on your own because I don't want to assume that everyone uses Ubuntu (as Tensorflow does).

## Data Format

In order to convert your data to TFRecords using `tfrecords.py` you need: A data folder that contains subdirectories such as `train`, `test` and `validation`. Each subdirectory contains images and a .csv file called `labels.csv` with the following fields: frame, xmin, xmax, ymin, ymax, class_id

## Acknowledgements

This project was inspired by the nightmare of the Tensorflow Object Detection API and by https://github.com/RomRoc/objdet_train_tensorflow_colab
