import os
import cv2
import numpy as np
import pylab as plt
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, ConvLSTM2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

data_path = r'C:\Users\mavc\Downloads\Den haag 2022\Selected images\Images with Rotterdam'


def load_images(directory, target_size=(100, 100)):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".tif"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Resize the image to the desired dimensions
            img = cv2.resize(img, target_size)

            images.append(img)
    return np.array(images)


# load thermal images
thermal_images = load_images(data_path)

# Normalize pixel values to be between 0 and 1
thermal_images = thermal_images / 255.0


# Labels
# Assuming you have corresponding labels/targets, load them accordingly
labels_int = np.array(range(0,50))
data_path_labels = r'C:\Users\mavc\Documents\Geomatics\thesis\Data QGIS project\Cluster ISODATA 20 classes Den Haag en Rotterdam (50 iterations).tif'
labels = cv2.imread(data_path_labels, cv2.IMREAD_GRAYSCALE)
plt.imshow(labels)

#-----------------------------------------------------------------------------------------------------------------------



# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(thermal_images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(X_train.shape)

# Reshape the data for ConvLSTM input
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], 1))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1], X_val.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], 1))

print(X_train)

# Define the ConvLSTM model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(1, X_train.shape[2], X_train.shape[3], 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define model checkpoint to save the best model during training
checkpoint = ModelCheckpoint('convlstm_model.h5', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), callbacks=[checkpoint])