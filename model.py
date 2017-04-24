import os
import sys
import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

BATCH_SIZE = 64
# correction for left and right camera images
camera_correction = 0.2

# reading data
data_folder = r'E:\windows_sim\sample_data'
csv_filepath = os.path.join(data_folder, 'driving_log.csv')

# prepare sample images.
# images from all cameras are used, and data is augmented using image flipping.
samples = []
with open(csv_filepath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        try:
            center_angle = float(line[3])
            samples.append([line[0].strip(), center_angle, 0])    # center camera
            samples.append([line[0].strip(), -center_angle, 1])    # center camera image flipped
            samples.append([line[1].strip(), center_angle+camera_correction, 0])    # left camera
            samples.append([line[1].strip(), -(center_angle+camera_correction), 1])    # left camera image flipped
            samples.append([line[2].strip(), center_angle-camera_correction, 0])    # right camera
            samples.append([line[2].strip(), -(center_angle-camera_correction), 1])    # right camera image flipped
        except:
            pass
        
train_samples, validation_samples = train_test_split(samples, test_size=0.1)

# data generator
def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                fname = os.path.join(data_folder, batch_sample[0])
                image = cv2.imread(fname)
                angle = batch_sample[1]
                if batch_sample[2]:
                    # flip image for data augmentation
                    image = np.fliplr(image)
                images.append(image)
                angles.append(angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield (X_train, y_train)
        
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

def resize_image(image):
    import tensorflow as tf  
    resize_image_size = (64, 64)
    return tf.image.resize_images(image, resize_image_size)            
            
# model
model = Sequential()

# Crop 50 pixels from the top of the image and 20 from the bottom
model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                     data_format='channels_last',
                     input_shape=(160, 320, 3)))

# normalization
model.add(Lambda(resize_image))
model.add(Lambda(lambda x: (x/255.0) - 0.5))
# conv layer 1
model.add(Conv2D(filters=32, 
            kernel_size=(5,5),
            activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv layer 2
model.add(Conv2D(filters=32, 
            kernel_size=(5,5),
            activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv layer 3
model.add(Conv2D(filters=64, 
            kernel_size=(5,5),
            activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
adam = Adam(lr=1e-4)

model.compile(loss="mse", optimizer=adam)

if "-s" in sys.argv:
    model.summary()
else:
    model.fit_generator(train_generator, 
                        steps_per_epoch=len(train_samples)/BATCH_SIZE, 
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples)/BATCH_SIZE, 
                        epochs=8)
    model.save("model.h5")
