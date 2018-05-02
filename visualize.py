import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import glob
from read_activations import display_activations, get_activations

#from new_model import CHARS


CHARS = '0123456789'

vali_datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
)

model = load_model("model-b.h5")

#dirs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dirs = ['3']
gen = vali_datagen.flow_from_directory('train_val_symbols/display/',
    classes=dirs,
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=1,
    class_mode=None,
    shuffle=False
)

#for i in range(len(model.layers)):
#  print(model.layers[i].name)

activations = get_activations(model, next(gen))
display_activations(activations)
