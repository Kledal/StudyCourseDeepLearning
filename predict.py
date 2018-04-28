import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import glob

#from new_model import CHARS


CHARS = '0123456789'

vali_datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
)

model = load_model("model.h5")

#dirs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dirs = ['3']
gen = vali_datagen.flow_from_directory('train_val_symbols/test/',
    classes=dirs,
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=1,
    class_mode=None,
    shuffle=False
)

probas = model.predict_generator(gen)
y_prob = probas[0]
max_idx = np.argmax(y_prob)
print(CHARS[max_idx])

#test_img = vali_datagen.
#y_prob = model.predict(np.array( [test_img] ))
#y_classes = y_prob.argmax(axis=-1)
#max_idx = np.argmax(y_prob)
#print(CHARS[max_idx])
