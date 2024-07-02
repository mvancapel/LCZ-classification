# LCZ-classification

This code provides a CNN with U-net that creates a Local Climate Zone classification from spatio-temporal thermal imagery. For the input data, a stack of thermal images is needed (can be created in QGIS), together with its corresponding mask (with class labels).

The data pre-processing steps can be found in preprocessing.py.

The U-net model including its evaluation metrics and analysis steps can be found in unet.py. The skeleton of the U-net was adopted from https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406.
