from __future__ import print_function
import math
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import keras as keras
from six.moves import range
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import numpy as np


# All the numbers, plus sign and space for padding.
CHARS = '0123456789' # -/.,ABCDEFGHIJKLMNOPQRSTUVXYZ'
MAX_EPOCHS = 100
NUM_CLASSES = len(CHARS)
IMAGES_PR_CLASS = 1

IMG_WIDTH = 64
IMG_HEIGHT = 64
INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT

filepath = 'model.h5'
# Define batch size
batch_size=100

samples_per_epoch = 4782/batch_size*2 #math.floor(NUM_CLASSES/batch_size)*4782
validation_steps = 1190/batch_size*2 #math.floor(NUM_CLASSES/batch_size)*10

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.1,
    rotation_range=10,
    shear_range=0.3,
    zoom_range=0.1,
    fill_mode='constant',
    cval=0,
)

vali_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    fill_mode='constant',
    cval=0,
)

print('Build model...')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

model = load_model("model.h5")

check_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                              save_best_only=True, save_weights_only=False, mode='auto', period=1)
early_stop = EarlyStopping(
    monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')

dirs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

model.fit_generator(
  train_datagen.flow_from_directory('train_val_symbols/train',
                                    classes=dirs,
                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                    color_mode='grayscale',
                                    batch_size=batch_size,
                                    class_mode='categorical'),
  verbose=1,
  epochs=MAX_EPOCHS,
  steps_per_epoch=samples_per_epoch,
  validation_data=vali_datagen.flow_from_directory('train_val_symbols/val',
                                                   classes=dirs,
                                                   target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                   color_mode='grayscale',
                                                   batch_size=batch_size,
                                                   class_mode='categorical'),
  validation_steps=validation_steps,
  callbacks=[check_point, early_stop]
)
