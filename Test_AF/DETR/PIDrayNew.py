import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd

from os.path import expanduser
from pathlib import Path
import pickle
import os

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img, load_img
from tensorflow.keras.applications.resnet import preprocess_input, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint

# Initilization

train_data_set_path = Path(os.environ["ICHOR_INPUT_DATASET"]) / "PIDray-Splitted/train"
test_data_set_path = Path(os.environ["ICHOR_INPUT_DATASET"]) / "PIDray-Splitted/test"
img_height = 224
img_width= 224
batch_size= 72

# Image generator for training set with data augmentation
train_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input, ##Maybe we do not need it
  rescale=1./255,
  rotation_range=45,
  width_shift_range=0.2,
  height_shift_range=0.2,
  brightness_range=[0.2,0.8],
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  )

# Image generator for testing. We only rescale the images in the test folder without data augmentation.
test_datagen = ImageDataGenerator(
  # preprocessing_function=preprocess_input, ##Maybe we do not need it
  rescale=1./255
  )

# Takes the path to a train directory & generates batches of augmented data.
train_generator = train_datagen.flow_from_directory(
  train_data_set_path,
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='categorical'
  )

# Takes the path to a test directory & generates batches.
test_generator = test_datagen.flow_from_directory(
  test_data_set_path,
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='categorical'
  )


# tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs)
pretrained_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='max', classes=12)

resnet_model = Sequential()
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(12, activation='softmax'))
# resnet_model.summary()

# 3. Train Model
resnet_model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

Path(os.environ["ICHOR_OUTPUT_DATASET"]).mkdir(exist_ok=True, parents=True)
Path(os.environ["ICHOR_LOGS"]).mkdir(exist_ok=True, parents=True)


# checkpoint_filepath = Path(os.environ["ICHOR_OUTPUT_DATASET"]) / "PIDrayCheckPoint.h5"
# callback = ModelCheckpoint(filepath=checkpoint_filepath, monitor='loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1)

# fit_generator is used when either we have a huge dataset to fit into our memory or when data augmentation needs to be applied.
# history = resnet_model.fit_generator(train_generator, steps_per_epoch=1217, epochs=1, callbacks=callback)
history = resnet_model.fit_generator(train_generator, steps_per_epoch=1217, epochs=1)


# # load json module
# import json

# # python dictionary with key value pairs
# dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}

# # create json object from dictionary
# json = json.dumps(dict)

fil = Path(os.environ["ICHOR_OUTPUT_DATASET"]) / "model.json"
x = Path(os.environ["ICHOR_OUTPUT_DATASET"]) / "model.h5"
# # open file for writing, "w" 
# f = open(fil,"w")

# # write json object to file
# f.write(json)

# # close file
# f.close()


# serialize model to JSON
model_json = resnet_model.to_json()

with open((fil), "w") as json_file:
    json_file.write(model_json)
    json_file.close()
# serialize weights to HDF5

resnet_model.save_weights(x)


# x = Path(os.environ["ICHOR_OUTPUT_DATASET"]) / "PIDrayCNN.h5"

# resnet_model.save(x)