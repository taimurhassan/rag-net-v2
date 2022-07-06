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

import nvidia_smi

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)
 
nvidia_smi.nvmlShutdown()

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 1:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)

print(len(physical_devices))

# class TrainConfig(TrainingConfig):
#     def __init__(self):
#         super().__init__()
#         self.data_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "pidray"
#         # self.data_dir = "C:/Users/mohda/Downloads/DETR Deep Learning/Main pidray/pidray"
#         self.data = DataConfig(data_dir=self.data_dir, img_dir="train", ann_file="annotations/xray_train.json")
#         self.image_size = (500, 500)
#         # self.normalized_method = 'tf_resnet'

# train_config = TrainConfig()
# train_iterator, class_names = load_coco_dataset(train_config, train_config.batch_size, augmentation=True)

# class ValidConfig(TrainingConfig):
#     def __init__(self):
#         super().__init__()
#         self.data_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "pidray"
#         # self.data_dir = "C:/Users/mohda/Downloads/DETR Deep Learning/Main pidray/pidray"
#         self.data = DataConfig(data_dir=self.data_dir, img_dir="easy", ann_file="annotations/xray_test_easy.json")
#         self.image_size = (500, 500)
#         # self.normalized_method = 'tf_resnet'

# valid_config = ValidConfig()
# valid_iterator, class_names = load_coco_dataset(valid_config, valid_config.batch_size, augmentation=None)

# for images, target_bbox, target_class in train_iterator:
#     print("images.shape", images.shape)
#     print("target_bbox.shape", target_bbox.shape)
#     print("target_class.shape", target_class.shape)

#     # Plot image
#     image = numpy_bbox_to_image(
#         np.array(images[0]),
#         np.array(target_bbox[0]),
#         labels=np.array(target_class[0]),
#         scores=None,
#         class_name=class_names,
#         config=train_config
#     )
#     plt.imshow(image)
#     break

# class_names

# detr = get_detr_model(train_config, include_top=False, nb_class=14, weights="detr", tf_backbone=True)
# # detr.summary()

# train_config.train_backbone = tf.Variable(True)
# train_config.train_transformers = tf.Variable(True)
# train_config.train_nlayers = tf.Variable(True)
# # train_config.nlayers_lr = tf.Variable(1e-3)
# train_config.backbone_lr = tf.Variable(1e-5)
# train_config.transformers_lr = tf.Variable(1e-4)
# train_config.nlayers_lr = tf.Variable(1e-4)

# optimzers = setup_optimizers(detr, train_config)

# detr.trainable = True
# # detr.summary()

# for epoch in range(1):
#     training.fit(detr, train_iterator, optimzers, train_config, epoch_nb=epoch, class_names=class_names)


# for valid_images, target_bbox, target_class in valid_iterator:    
#     m_outputs = detr(valid_images, training=False)
#     predicted_bbox, predicted_labels, predicted_scores = get_model_inference(m_outputs, valid_config.background_class, bbox_format="xy_center")
#     print(predicted_bbox)
#     print(predicted_labels)
#     print(predicted_scores)

#     result = numpy_bbox_to_image(
#         np.array(valid_images[0]),
#         np.array(predicted_bbox),
#         np.array(predicted_labels),
#         scores=np.array(predicted_scores),
#         class_name=class_names, 
#         config=valid_config
#     )
#     plt.imshow(result)
#     plt.show()
#     break

# detr.save_weights("DETR_300_weights.ckpt")