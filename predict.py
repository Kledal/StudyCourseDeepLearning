import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import glob

#from new_model import CHARS

CHARS = '0123456789-ABCDEFGHIJKLMNPQRSTUVXYZ'

vali_datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
)

model = load_model("model-all.h5")

#dirs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

matrix = []

dirs = list(CHARS)
for folder in dirs:
  gen = vali_datagen.flow_from_directory('train_val_symbols/val/',
      classes=[folder],
      target_size=(64, 64),
      color_mode='grayscale',
      batch_size=1,
      class_mode=None,
      shuffle=False
  )
  
  probas = model.predict_generator(gen)
  failed = 0
  
  for i in range(len(probas)):
      y_prob = probas[i]
      max_idx = np.argmax(y_prob)
      predict = CHARS[max_idx]

      matrix.append([folder, predict])

      if predict != folder:
          failed = failed +1
  
  print("Class " + folder + ": Total vali: " + str(len(probas)) + ", failed: " + str(failed))

np.save("true-predict", np.array(matrix))

#test_img = vali_datagen.
#y_prob = model.predict(np.array( [test_img] ))
#y_classes = y_prob.argmax(axis=-1)
#max_idx = np.argmax(y_prob)
#print(CHARS[max_idx])
