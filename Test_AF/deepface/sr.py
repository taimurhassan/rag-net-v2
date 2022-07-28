import os
import tqdm
import pandas as p
from pathlib import Path
import pickle
from matplotlib import image
import cv2
import numpy as np
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
from super_image import PanModel, ImageLoader


data_set_path = Path(os.environ["ICHOR_INPUT_DATASET"]) / "droneSURF/Active_Even_L1/1/"


model = PanModel.from_pretrained('eugenesiow/pan', scale=4)


fil = '/mnt/datasets/rag-net-v2-0c6f96b8050c43fd-outputs/output/SR/'

#!/usr/bin/python
from os import listdir
from PIL import Image as PImage

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)
#         print(image)
        if(os.path.exists(fil+image)):
            print(image + " Exisits")
            pass
        else:
            inputs = ImageLoader.load_image(img)
            preds = model(inputs)
            ImageLoader.save_image(preds, fil + image)
            print("Image ", image, " is Converted")
    return loadedImages


# your images in an array
imgs = loadImages(data_set_path)