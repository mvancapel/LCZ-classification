import rasterio
import time
import os
import numpy as np  # for arrays
import tensorflow as tf
import matplotlib.pyplot as plt
import tifffile  # The input images are in .tiff format and can be parsed using this library
import tensorflow.keras as keras
from PIL import Image
import matplotlib.patches as mpatches
import copy
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from matplotlib.colors import from_levels_and_colors

# for bulding and running deep learning model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.regularizers import l2

import matplotlib.colors
import pandas as pd

def LoadData(path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively
    """
    # Read the images folder like a list
    mask_dataset = os.listdir(path2)
    image_dataset = os.listdir(path1)

    # Make a list for images and masks filenames
    orig_img = []
    mask_img = []
    for file in image_dataset:
        if file.endswith('.tif'):
            orig_img.append(file)
    for file in mask_dataset:
        if file.endswith('.tif'):
            mask_img.append(file)

    # Sort the lists to get both of them in same order
    orig_img.sort()
    mask_img.sort()
    return orig_img, mask_img


def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    """
    # Pull the relevant dimensions for image and mask
    m = len(img)  # number of images
    i_h, i_w, i_c = target_shape_img  # pull height, width, and channels of image
    m_h, m_w, m_c = target_shape_mask  # pull height, width, and channels of mask
    # print(target_shape_mask)

    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)

    # Resize images and masks
    for file in img:
        # convert image into an array of desired shape (10 channels)
        index = img.index(file)
        path = os.path.join(path1, file)
        image = tifffile.imread(path)
        # image = image.resize((i_h,i_w))
        image = np.reshape(image, (i_h, i_w, i_c))

        # NEED TO UPDATE NORMALIZATION: e.g. get max value of each band
        # image = image/256
        X[index] = image

        # convert mask into an array of desired shape (1 channel)
        single_mask_ind = mask[index]
        # print(single_mask_ind)
        # print(path2)
        path = os.path.join(path2, single_mask_ind)
        # print(path)
        mask_arr = Image.open(path)
        mask_arr = mask_arr.resize((m_h, m_w))
        mask_arr = np.reshape(mask_arr, (m_h, m_w, m_c))
        # mask_resized = np.expand_dims(mask_arr, axis=2)
        y[index] = mask_arr

    return X, y


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, L2factor=0.1, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool,
    relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer
    along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization
    # Proper initialization prevents from the problem
    # of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that
    # the output has the same height and width (hence, is not reduced in size)
    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_regularizer=l2(L2factor),
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_regularizer=l2(L2factor),
                  kernel_initializer='HeNormal')(conv)

    # Batch Normalization will normalize the output of the last layer based
    # on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the l oss and gradient
    # computation to shrink the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels
    # Pooling has been kept as optional as the last encoder layer
    # does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output
    # computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer
    # to prevent information loss during transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection


def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size
    up = Conv2DTranspose(
        n_filters,
        (3, 3),  # Kernel size
        strides=(2, 2),
        padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)

    # Add 2 Conv Layers with relu activation and HeNormal initialization
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    return conv


def UNetCompiled(input_size=(32, 32, 25), n_filters=32, n_classes=26):
    """"
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, L2factor=0.1, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2, dropout_prob=0, L2factor=0.1, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4, dropout_prob=0, L2factor=0.1, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters * 8, dropout_prob=0.3, L2factor=0.1, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters * 16, dropout_prob=0.3, L2factor=0.1, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1], n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    # sigmoid converts negative values to 0 (predictions between 0-1)
    conv10 = Conv2D(n_classes, 1, padding='same', activation=activation)(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

path1 = './data/satellite/'
path2 = './data/mask/'
img, mask = LoadData(path1, path2)

# View an example of image and corresponding mask
i = 9
img_view  = tifffile.imread(path1 + img[i])
mask_view = tifffile.imread(path2 + mask[i])
print(img_view.shape)
print(mask_view.shape)
fig, arr = plt.subplots(1, 2, figsize=(10, 10))
# just view band 1 of satellite image
arr[0].imshow(img_view[:,:,1])
arr[0].set_title('Image '+ str(i))

arr[1].imshow(mask_view)
arr[1].set_title('Masked Image '+ str(i))

plt.show()

# Define the desired shape: depends on number of channels and size
# UPDATE WITH DIFFERENT NUMBER OF CLASSES AND CHANNELS
n_classes = 26 # 1-17 & 25
n_channels = 25 # Number of bands in stack
size = 32

target_shape_img = [size, size, n_channels]
target_shape_mask = [size, size, 1]

X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2)

# QC the shape of output and classes in output dataset
print("X Shape:", X.shape)
print("Y shape:", y.shape)
# There are 3 classes : background, pet, outline
print(np.unique(y))

count_label = np.count_nonzero(y < 18 )
count_zero = np.count_nonzero(y == 25)

total_pixels = np.count_nonzero(y)
total_labelled = count_label
percent_labelled = total_labelled / total_pixels
print(f'Total labelled pixels:{total_labelled}\nPercent labelled pixels: {percent_labelled}')

# Use scikit-learn's function to split the dataset
random=42
# save 10% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random)
X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(img, mask, test_size=0.1, random_state=random)
# Also create a validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=random)
X_train_file, X_valid_file, y_train_file, y_valid_file = train_test_split(img, mask, test_size=0.1, random_state=random)

count_label = np.count_nonzero(y_test < 18 )
count_zero = np.count_nonzero(y_test == 25)

total_pixels = np.count_nonzero(y_test)
total_labelled = count_label
percent_labelled = total_labelled / total_pixels
print(f'test sample randomization: {random}\n test sample size:{total_labelled}\n test percent_labelled: {percent_labelled}')

band = 1
for i in range(X_train.shape[3]):
    flattened = X_train[:, :, :, i].flatten()

    q_1 = np.quantile(flattened, 0.1)
    q_1 = tf.convert_to_tensor(q_1)

    q_99 = np.quantile(flattened, 0.99)
    q_99 = tf.convert_to_tensor(q_99)

    max_ = flattened.max()
    max_ = tf.convert_to_tensor(max_)

    min_ = flattened.min()
    min_ = tf.convert_to_tensor(min_)
    img = 1
    band += 1
    # update this band for each image
    for im in range(X_train.shape[0]):

        # min and max
        X_train[im, :, :, i] = (X_train[im, :, :, i] - min_) * ((1 - 0) / (max_ - min_)) + 0
        img += 1
    for im in range(X_test.shape[0]):

        # min and max
        X_test[im, :, :, i] = (X_test[im, :, :, i] - min_) * ((1 - 0) / (max_ - min_)) + 0
        img += 1
    for im in range(X_valid.shape[0]):
        # min and max
        X_valid[im, :, :, i] = (X_valid[im, :, :, i] - min_) * ((1 - 0) / (max_ - min_)) + 0
        img += 1

unet = UNetCompiled(input_size=(size, size, n_channels), n_filters=32, n_classes=n_classes)
unet.summary()

# There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
# Ideally, try different options to get the best accuracy
unet.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

results = unet.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_valid, y_valid))

# High Bias is a characteristic of an underfitted model and we would observe low accuracies for both train and validation set
# High Variance is a characterisitic of an overfitted model and we would observe high accuracy for train set and low for validation set
# To check for bias and variance plit the graphs for accuracy
# I have plotted for loss too, this helps in confirming if the loss is decreasing with each iteration - hence, the model is optimizing fine

fig, axis = plt.subplots(1, 2, figsize=(20, 5))
axis[0].plot(results.history["loss"], color='r', label = 'train loss')
axis[0].plot(results.history["val_loss"], color='b', label = 'dev loss')
axis[0].set_title('Loss Comparison')
axis[0].legend()
axis[1].plot(results.history["accuracy"], color='r', label = 'train accuracy')
axis[1].plot(results.history["val_accuracy"], color='b', label = 'dev accuracy')
axis[1].set_title('Accuracy Comparison')
axis[1].legend()

plt.show()

unet.evaluate(X_valid, y_valid)

# Results of Validation Dataset
def VisualizeResults(index):
    cmap, norm = from_levels_and_colors([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 25], ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'lime', 'peachpuff', 'magenta', 'lavender', 'maroon', 'navy', 'olive', 'mediumvioletred', 'black'])
    img = X_valid[index]
    img = img[np.newaxis, ...]
    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    fig, arr = plt.subplots(1, 3, figsize=(15, 15))
    arr[0].imshow(X_valid[index][:,:,1])
    arr[0].set_title('Image')
    arr[1].imshow(y_valid[index,:,:,0], cmap = cmap, norm = norm)
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask[:,:,0], cmap = cmap, norm = norm)
    arr[2].set_title('Predicted Masked Image ')


# # Add any index to contrast the predicted mask with actual mask
index = 1
VisualizeResults(index)

