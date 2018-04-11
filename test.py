from __future__ import print_function
from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import keras as keras
from six.moves import range

from image import generateImage

# All the numbers, plus sign and space for padding.
CHARS = '0123456789-ABCDEFGHIJKLMNOPQRSTUVXYZ'
MAX_EPOCHS = 32
NUM_CLASSES = len(CHARS)
IMG_WIDTH = 64
IMG_HEIGHT = 64
INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.1,
    rotation_range=10,
    shear_range=0.3,
    zoom_range=0.1,
)

x = [] # Images
y = [] # Label

for idx in range(NUM_CLASSES):
  for i in range(1):
    label = CHARS[idx]
    img = generateImage(label)
    newImg = img_to_array(img)
    #newImg = newImg.reshape((1,) + newImg.shape)
    #print(newImg)
    x.append(newImg)
    y.append(idx)

y = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

x = np.array(x)
y = np.array(y)

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Build model...')
model = Sequential()
model.add(layers.Dense(100, activation='relu', input_shape=(INPUT_SIZE, )))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation='relu'))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=10)

# Fit 
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
batch_size=32
#model.fit_generator(datagen.flow(x_train, y_train, 
#                    batch_size=batch_size),
#                    steps_per_epoch=len(x_train) / batch_size, 
#                    epochs=MAX_EPOCHS)
# here's a more "manual" example
for e in range(MAX_EPOCHS):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
        model.fit(x_batch.reshape(batch_size, INPUT_SIZE), y_batch)
        batches += 1
        if batches >= len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

score = model.evaluate(x_val.reshape(len(x_val), INPUT_SIZE), y_val, batch_size=batch_size, verbose=1)
print("Score: " + str(score))
