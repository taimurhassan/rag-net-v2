import os
import tqdm
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception
from pathlib import Path
import pandas as p

task_dataset_folder = os.environ.get("ICHOR_INPUT_DATASET", "/COMPASS-XP")

train_dir = pickle.load(open(os.path.join(task_dataset_folder, "task_dataset.pkl"), "rb"))


# train_dir =os.path.join(original_dataset_dir+"\\Training")
# test_dir =os.path.join(original_dataset_dir+"\\Testing")


n=1

# conv_base =DenseNet201(input_shape = (120, 120, 3), # Shape of our images
#                   include_top = False, # Leave out the last fully connected layer
#                   weights = 'imagenet')

# conv_base = ResNet50(input_shape=(120, 120, 3),  # Shape of our images
#                      include_top=False,  # Leave out the last fully connected layer
#                      weights='imagenet')

# conv_base = VGG19(input_shape = (120, 120, 3), # Shape of our images
#                   include_top = False, # Leave out the last fully connected layer
#                   weights = 'imagenet')

conv_base = VGG16(input_shape=(120, 120, 3),  # Shape of our images
                  include_top=False,  # Leave out the last fully connected layer
                  weights='imagenet')

# conv_base = InceptionV3(input_shape=(120, 120, 3),  # Shape of our images
#                   include_top=False,  # Leave out the last fully connected layer
#                   weights='imagenet')

# conv_base = InceptionResNetV2(input_shape=(120, 120, 3),  # Shape of our images
#                   include_top=False,  # Leave out the last fully connected layer
#                   weights='imagenet')

# conv_base = Xception(input_shape=(120, 120, 3),  # Shape of our images
#                   include_top=False,  # Leave out the last fully connected layer
#                   weights='imagenet')

conv_base.trainable = False


# conv_base.trainable = True
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1' and layer.name == 'block5_conv2' and layer.name == 'block5_conv3'  and layer.name == 'block4_conv1' and layer.name == 'block4_conv2' and layer.name == 'block4_conv3':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False


model = models.Sequential()

model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(n, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.01, name="Adam", ), metrics=['acc'])



train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir, target_size=(120, 120),batch_size= 32,
                                                        class_mode='binary')

# Test_generator = val_datagen.flow_from_directory(test_dir, target_size=(120, 120),
#                                                            class_mode='binary')


history = model.fit_generator(train_generator, epochs=100)

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

print(model.evaluate(train_generator))
