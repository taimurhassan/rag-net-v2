import sys
# Set the path to the repository here
# sys.path.append("C:/Users/mohda/Documents/GitHub/rag-net-v2/Test_AF/DETR/detr_tf")

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os.path import expanduser
from pathlib import Path
import pandas as p
import pickle
from matplotlib import image
import os
import detr_tf
from detr_tf.data import load_coco_dataset
from detr_tf.networks.detr import get_detr_model
from detr_tf.training_config import TrainingConfig, DataConfig
from detr_tf.inference import numpy_bbox_to_image
from detr_tf.optimizers import setup_optimizers
from detr_tf import training
from detr_tf.inference import get_model_inference, numpy_bbox_to_image

 
class TrainConfig(TrainingConfig):
    def __init__(self):
        super().__init__()
        self.data_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "pidray"
        # self.data_dir = "C:/Users/mohda/Downloads/DETR Deep Learning/Main pidray/pidray"
        self.data = DataConfig(data_dir=self.data_dir, img_dir="train", ann_file="annotations/xray_train.json")
        self.image_size = (500, 500)
        # self.normalized_method = 'tf_resnet'

train_config = TrainConfig()
train_iterator, class_names = load_coco_dataset(train_config, train_config.batch_size, augmentation=True)

class ValidConfig(TrainingConfig):
    def __init__(self):
        super().__init__()
        self.data_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "pidray"
        # self.data_dir = "C:/Users/mohda/Downloads/DETR Deep Learning/Main pidray/pidray"
        self.data = DataConfig(data_dir=self.data_dir, img_dir="easy", ann_file="annotations/xray_test_easy.json")
        self.image_size = (500, 500)
        # self.normalized_method = 'tf_resnet'

valid_config = ValidConfig()
valid_iterator, class_names = load_coco_dataset(valid_config, valid_config.batch_size, augmentation=None)

for images, target_bbox, target_class in train_iterator:
    print("images.shape", images.shape)
    print("target_bbox.shape", target_bbox.shape)
    print("target_class.shape", target_class.shape)

    # Plot image
    image = numpy_bbox_to_image(
        np.array(images[0]),
        np.array(target_bbox[0]),
        labels=np.array(target_class[0]),
        scores=None,
        class_name=class_names,
        config=train_config
    )
    plt.imshow(image)
    break

class_names

detr = get_detr_model(train_config, include_top=False, nb_class=14, weights="detr", tf_backbone=True)
# detr.summary()

train_config.train_backbone = tf.Variable(True)
train_config.train_transformers = tf.Variable(True)
train_config.train_nlayers = tf.Variable(True)
# train_config.nlayers_lr = tf.Variable(1e-3)
train_config.backbone_lr = tf.Variable(1e-5)
train_config.transformers_lr = tf.Variable(1e-4)
train_config.nlayers_lr = tf.Variable(1e-4)

optimzers = setup_optimizers(detr, train_config)

detr.trainable = True
# detr.summary()

for epoch in range(1):
    training.fit(detr, train_iterator, optimzers, train_config, epoch_nb=epoch, class_names=class_names)
