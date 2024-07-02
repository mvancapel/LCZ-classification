import rasterio
import time
import os
import numpy as np  # for arrays
import tensorflow as tf
import matplotlib.pyplot as plt
import tifffile  # The input images are in .tiff format and can be parsed using this library
import tensorflow.keras as keras
from PIL import Image
from matplotlib.colors import from_levels_and_colors
from tensorflow.keras import backend as K

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
from tensorflow.keras.callbacks import Callback

import sklearn
import matplotlib.colors
import pandas as pd


def LoadData(path1, path2):
    """
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


def UNetCompiled(input_size, n_filters=32, n_classes=11):
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

    conv10 = Conv2D(n_classes, 1, padding='same', activation = 'softmax')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


path1 = './data_all/satellite/'
path2 = './data_all/mask/'
img, mask = LoadData(path1, path2)

# View an example of image and corresponding mask

# i = 9
# img_view = tifffile.imread(path1 + img[i])
# mask_view = tifffile.imread(path2 + mask[i])
# print(img_view.shape)
# print(mask_view.shape)
# fig, arr = plt.subplots(1, 2, figsize=(10, 10))
# # just view band 1 of satellite image
# arr[0].imshow(img_view[:, :, 1])
# arr[0].set_title('Image ' + str(i))
#
# arr[1].imshow(mask_view)
# arr[1].set_title('Masked Image ' + str(i))
# plt.show()

# Define the desired shape: depends on number of channels and size
# UPDATE WITH DIFFERENT NUMBER OF CLASSES AND CHANNELS
n_classes = 11  # 1-17
n_channels = 46  # Number of bands in stack
size = 64

target_shape_img = [size, size, n_channels]
target_shape_mask = [size, size, 1]

X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2)

# QC the shape of output and classes in output dataset
print("X Shape:", X.shape)
print("Y shape:", y.shape)
# Print the unique classes
print(np.unique(y))

count_label = np.count_nonzero(y < 11)
count_zero = np.count_nonzero(y == 25)
count_1 = np.count_nonzero(y == 1)
count_2 = np.count_nonzero(y == 2)
count_3 = np.count_nonzero(y == 3)
count_4 = np.count_nonzero(y == 4)
count_5 = np.count_nonzero(y == 5)
count_6 = np.count_nonzero(y == 6)
count_7 = np.count_nonzero(y == 7)
count_8 = np.count_nonzero(y == 8)
count_9 = np.count_nonzero(y == 9)
count_10 = np.count_nonzero(y == 10)


total_pixels = np.count_nonzero(y)
total_labelled = count_label
percent_labelled = total_labelled / total_pixels
print(f'Total labelled pixels:{total_labelled}\nPercent labelled pixels: {percent_labelled}')
print(f'number of pixels class 1: {count_1}, {count_1/total_pixels}')
print(f'number of pixels class 2: {count_2}, {count_2/total_pixels}')
print(f'number of pixels class 3: {count_3}, {count_3/total_pixels}')
print(f'number of pixels class 4: {count_4}, {count_4/total_pixels}')
print(f'number of pixels class 5: {count_5}, {count_5/total_pixels}')
print(f'number of pixels class 6: {count_6}, {count_6/total_pixels}')
print(f'number of pixels class 7: {count_7}, {count_7/total_pixels}')
print(f'number of pixels class 8: {count_8}, {count_8/total_pixels}')
print(f'number of pixels class 9: {count_9}, {count_9/total_pixels}')
print(f'number of pixels class 10: {count_10}, {count_10/total_pixels}')

# Use scikit-learn's function to split the dataset
random = 42
# save 10% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random)
X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(img, mask, test_size=0.15, random_state=random)
# Also create a validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.17647, random_state=random) # 0.17647 * 0.85 = 0.15
X_train_file, X_valid_file, y_train_file, y_valid_file = train_test_split(img, mask, test_size=0.17647, random_state=random) # 0.17647 * 0.85 = 0.15

count_label = np.count_nonzero(y_test < 11)
count_zero = np.count_nonzero(y_test == 25)

total_pixels = np.count_nonzero(y_test)
total_labelled = count_label
percent_labelled = total_labelled / total_pixels
print(
    f'test sample randomization: {random}\n test sample size:{total_labelled}\n test percent_labelled: {percent_labelled}')

# convert no-data value to 0 ???
for i in range(y_train.shape[0]):
    y_train[i, :, :] = tf.where(y_train[i, :, :] == 25, 0, y_train[i, :, :])


# Normalize the satellite data
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


def custom_accuracy(y_true, y_pred):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    y_true = tf.reshape(y_true, shape=(-1, 64, 64))
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')
    # apply mask to exclude where y_true equals 25
    y_pred = tf.where(y_true == 25, tf.zeros_like(y_pred), y_pred)
    true_positives = K.sum((K.cast(K.equal(y_true, y_pred), dtype=tf.float32)))
    all_predictions = K.cast((K.sum(tf.where(y_true == 0, 0, 1))), dtype=tf.float32)
    # returns accuracy
    return true_positives / all_predictions

# def recall(y_true, y_pred):
#     y_pred = tf.math.argmax(y_pred, axis=-1)
#     y_true = tf.reshape(y_true, shape=(-1, 32, 32))
#     y_true = tf.cast(y_true, 'float32')
#     y_pred = tf.cast(y_pred, 'float32')
#     # apply mask to exclude where y_true equals 25
#     y_pred = tf.where(y_true == 25, tf.zeros_like(y_pred), y_pred)
#     recall = []
#     for i in range(1, 18):
#         mask_y_true = tf.equal(y_true, i)
#         mask_y_pred = tf.equal(y_true, i)
#         true_positives = K.sum(K.cast(tf.logical_and(mask_y_true, mask_y_pred), dtype=tf.float32))
#         possible_positives = K.sum(K.cast(mask_y_true, dtype=tf.float32))
#         recall_i = true_positives / (possible_positives + K.epsilon())
#         recall.append(recall_i)
#     recall = tf.stack(recall)
#     return recall
#
#
# def precision(y_true, y_pred):
#     y_pred = tf.math.argmax(y_pred, axis=-1)
#     y_true = tf.reshape(y_true, shape=(-1, 32, 32))
#     y_true = tf.cast(y_true, 'float32')
#     y_pred = tf.cast(y_pred, 'float32')
#     # apply mask to exclude where y_true equals 25
#     y_pred = tf.where(y_true == 25, tf.zeros_like(y_pred), y_pred)
#     precision = []
#     for i in range(1, 18):
#         mask_y_true = tf.equal(y_true, i)
#         mask_y_pred = tf.equal(y_true, i)
#         true_positives = K.sum(K.cast(tf.logical_and(mask_y_true, mask_y_pred), dtype=tf.float32))
#         predicted_positives = K.sum(K.cast(K.equal(y_pred, i), dtype=tf.float32))
#         precision_i = true_positives / (predicted_positives + K.epsilon())
#         precision.append(precision_i)
#     precision = tf.stack(precision)
#     print('precision = ', precision)
#     return precision

class PrecisionCallback(Callback):
    def __init__(self, validation_data=None, X_test=None, y_test=None):
        super(PrecisionCallback, self).__init__()
        self.validation_data = validation_data
        self.X_test = X_test
        self.y_test = y_test
        self.precisions_fit = []
        self.precisions_evaluate = []

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is not None: # Handle model.fit
            val_x, val_y = self.validation_data
            y_pred = unet.predict(val_x)
            y_pred_classes = np.argmax(y_pred, axis=-1)
            y_true = tf.reshape(val_y, shape=(-1, 64, 64))
            precision = []
            for i in range(1, 11):
                mask_y_true = np.equal(y_true, i)
                mask_y_pred = np.equal(y_pred_classes, i)
                true_positives = np.sum(tf.logical_and(mask_y_true, mask_y_pred))
                predicted_positives = np.sum(mask_y_pred)
                precision_i = true_positives / (predicted_positives + 1e-7) # Adding a small epsilon to avoid division by zero
                precision.append(precision_i)
            precision = np.array(precision)
            self.precisions_fit.append(precision)
            # print('Precision per class at epoch {} (fit):'.format(epoch + 1), precision)
        else: # Handle model.evaluate
            y_pred = unet.predict(self.X_test)
            y_pred_classes = np.argmax(y_pred, axis=-1)
            y_true = tf.reshape(self.y_test, (-1, 64, 64))
            precision = []
            for i in range(1, 11):
                mask_y_true = np.equal(y_true, i)
                mask_y_pred = np.equal(y_pred_classes, i)
                true_positives = np.sum(tf.logical_and(mask_y_true, mask_y_pred))
                predicted_positives = np.sum(mask_y_pred)
                precision_i = true_positives / (predicted_positives + 1e-7)  # Adding a small epsilon to avoid division by zero
                precision.append(precision_i)
            precision = np.array(precision)
            self.precisions_evaluate.append(precision)
            # print('Precision per class at epoch {} (evaluation):'.format(epoch + 1), precision)


class RecallCallback(Callback):
    def __init__(self, validation_data=None, X_test=None, y_test=None):
        super(RecallCallback, self).__init__()
        self.validation_data = validation_data
        self.X_test = X_test
        self.y_test = y_test
        self.recalls_fit = []
        self.recalls_evaluate = []

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is not None: # Handle model.fit
            val_x, val_y = self.validation_data
            y_pred = unet.predict(val_x)
            y_pred_classes = np.argmax(y_pred, axis=-1)
            y_true = tf.reshape(val_y, shape=(-1, 64, 64))
            recall = []
            for i in range(1, 11):
                mask_y_true = np.equal(y_true, i)
                mask_y_pred = np.equal(y_pred_classes, i)
                true_positives = np.sum(tf.logical_and(mask_y_true, mask_y_pred))
                possible_positives = np.sum(mask_y_true)
                recall_i = true_positives / (possible_positives + 1e-7)  # Adding a small epsilon to avoid division by zero
                recall.append(recall_i)
            recall = np.array(recall)
            self.recalls_fit.append(recall)
            # print('Recall per class at epoch {} (fit):'.format(epoch + 1), recall)
        elif self.X_test is not None and self.y_test is not None:  # Handle model.evaluate
            print('hoi')
            y_pred = unet.predict(self.X_test)
            y_pred_classes = np.argmax(y_pred, axis=-1)
            y_true = tf.reshape(self.y_test, (-1, 64, 64))
            recall = []
            for i in range(1, 11):
                mask_y_true = np.equal(y_true, i)
                mask_y_pred = np.equal(y_pred_classes, i)
                true_positives = np.sum(np.logical_and(mask_y_true, mask_y_pred))
                possible_positives = np.sum(mask_y_true)
                recall_i = true_positives / (possible_positives + 1e-7)  # Adding a small epsilon to avoid division by zero
                recall.append(recall_i)
            recall = np.array(recall)
            self.recalls_evaluate.append(recall)
            # print('Recall per class at epoch {} (evaluation):'.format(epoch + 1), recall)

# Define the layers for the model, given the input image size
unet = UNetCompiled(input_size=(size, size, n_channels), n_filters=32, n_classes=n_classes)
unet.summary()

# Compile: Configuring the model for training
unet.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             metrics=[custom_accuracy, 'accuracy', 'sparse_categorical_accuracy'])

precision_callback_fit = PrecisionCallback(validation_data=(X_valid, y_valid))
recall_callback_fit = RecallCallback(validation_data=(X_valid, y_valid))


class_weight = {0: 0.,
                1: 1.,
                2: 1.,
                3: 1.,
                4: 1.,
                5: 1.,
                6: 1.,
                7: 1.,
                8: 1.,
                9: 1.,
                10: 1.}

# Fit: Training the model or a chosen number of epochs
results = unet.fit(X_train, y_train, batch_size=32, epochs=100, class_weight=class_weight, validation_data=(X_valid, y_valid), callbacks = [recall_callback_fit, precision_callback_fit])

precision = precision_callback_fit.precisions_fit
recall = recall_callback_fit.recalls_fit

predictions = unet.predict(X_train)
probability = np.max(predictions, axis=-1)
predictions = np.argmax(predictions, axis=-1)
print('predictions = ', predictions)
print('probability = ', probability)

predict_array = np.reshape(predictions, (98, 64, 64))
prob_array = np.reshape(probability, (98, 64, 64))
for i in range(1, 23):
    with rasterio.open(f'./data_all/mask/{i}.tif', 'r') as src:
        output_predict = f'./data_all/predictions/predict_{i}.tif'
        output_prob = f'./data_all/predictions/prob_{i}.tif'
        label_meta = src.meta.copy()
        print(label_meta)

    with rasterio.open(output_predict, 'w', **label_meta) as dst:
        dst.write(predict_array[i-1], 1)

    label_meta.update({
        'dtype': 'float32'
    })
    with rasterio.open(output_prob, 'w', **label_meta) as dst:
        dst.write(prob_array[i-1], 1)

# Calculate the average probability across all classes for each pixel
average_prob = np.mean(prob_array, axis=0)

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(average_prob, cmap='hot', aspect='auto')
plt.colorbar(label='Average Probability')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Average Probability Across All Classes (train data)')
plt.show()


# Calculate the probability for each class across all pixels
class_probabilities = []
for class_num in range(1, 11):
    class_mask = (predict_array == class_num)
    class_prob = prob_array[class_mask]
    class_probabilities.append(class_prob)

# Create a box plot for each class
plt.figure(figsize=(10, 6))
plt.boxplot(class_probabilities, labels=np.arange(1, 11))
plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Probability Distribution per Class (train data)')
plt.show()

# Evaluate: Returns the loss value & metrics values for the model in test mode. Computation is done in batches.
test_metrics = unet.evaluate(X_test, y_test)

precision_callback_evaluate = PrecisionCallback(X_test=X_test, y_test=y_test)
recall_callback_evaluate = RecallCallback(X_test=X_test, y_test=y_test)
precision_callback_evaluate.on_epoch_end(epoch=None)
recall_callback_evaluate.on_epoch_end(epoch=None)

recall_test = recall_callback_evaluate.recalls_evaluate
precision_test = precision_callback_evaluate.precisions_evaluate

f1_test = []
for i in range(len(precision_test)):
    f1 = 2 * ((precision_test[i] * recall_test[i]) / (precision_test[i] + recall_test[i] + K.epsilon()))
    f1_test.append(f1)

print('recall_evaluate = ', recall_test)
print('test_metrics = ', test_metrics)
print('precision test =', precision_callback_evaluate.precisions_evaluate)

loss = results.history['loss']
val_loss = results.history['val_loss']

accuracy = results.history['custom_accuracy']
val_accuracy = results.history['val_custom_accuracy']

# Calculate F1_score with precision and recall per epoch during training
F1_score = []
for i in range(len(precision)):
    f1_score_i = []
    for j in range(len(precision[i])):
        f1 = 2 * ((precision[i][j] * recall[i][j]) / (precision[i][j] + recall[i][j] + K.epsilon()))
        f1_score_i.append(f1)
    F1_score.append(f1_score_i)
print(F1_score)
transposed_F1_score = list(zip(*F1_score))


#Calculate micro F1 score
micro_precision = np.sum(precision, axis=1) / len(precision[0])
micro_recall = np.sum(recall, axis=1) / len(recall[0])

micro_f1_per_epoch = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-7)  # Adding a small epsilon to avoid division by zero

print(results.history.keys())

fig, axis = plt.subplots(1, 4, figsize=(20, 5))
axis[0].plot(loss, color='r', label='train loss')
axis[0].plot(val_loss, color='b', label='val loss')
axis[0].axhline(y = test_metrics[0], color='g', label= 'test loss')
#axis[0].plot(test_metrics[0])
axis[0].set_title('Loss Comparison')
axis[0].legend()
axis[0].set_xlabel('Epochs')
axis[1].plot(results.history['custom_accuracy'], color='r', label='train accuracy')
axis[1].plot(results.history['val_custom_accuracy'], color='b', label='val accuracy')
axis[1].axhline(y = test_metrics[1], color='g', label='test accuracy')
axis[1].set_title('Accuracy Comparison')
axis[1].legend()
axis[1].set_xlabel('Epochs')
for index,line in enumerate(transposed_F1_score):
    axis[2].plot(line, label=f'Class={index+1}')
axis[2].legend()
axis[2].set_title('F1 score per class')
axis[2].set_xlabel('Epochs')
axis[3].plot(micro_f1_per_epoch, color='r', label='train micro f1')
axis[3].axhline(y = micro_f1_per_epoch[99], color ='g', label='test micro f1')
axis[3].set_title('F1 score')
axis[3].set_xlabel('Epochs')
axis[3].legend()

plt.show()

print('test_accuracy = ', test_metrics[1], 'test_f1_score = ', micro_f1_per_epoch[99], 'test_f1_score_per_class = ', f1_test)
# ---------------------------------------------------------------------------------------------------------------------------------------------
# Results of Validation Dataset
def VisualizeResults(index):
    cmap, norm = from_levels_and_colors([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 25],
                                        ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan',
                                         'lime', 'peachpuff', 'magenta', 'lavender', 'maroon', 'navy', 'olive',
                                         'mediumvioletred', 'black'])
    img = X_valid[index]
    img = img[np.newaxis, ...]
    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    fig, arr = plt.subplots(1, 3, figsize=(15, 15))
    arr[0].imshow(X_valid[index][:, :, 1])
    arr[0].set_title('Image')
    arr[1].imshow(y_valid[index, :, :, 0], cmap=cmap, norm=norm)
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask[:, :, 0], cmap=cmap, norm=norm)
    arr[2].set_title('Predicted Masked Image ')


# # Add any index to contrast the predicted mask with actual mask
index = 8
VisualizeResults(index)
