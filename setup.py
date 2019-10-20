#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='objdet',
      version='1.0',
      packages=find_packages(),
      install_requires=['pandas',
                        'numpy',
                        'matplotlib',
                        'tensorflow==1.15.0'],
      description='Object Detection in Tensorflow',
      author='Manuel Cherep',
      author_email='manuel.cherep@epfl.ch',
      url='https://github.com/mcherep/objdet'
      )
