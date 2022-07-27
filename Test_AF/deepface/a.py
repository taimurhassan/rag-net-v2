import os
import tqdm
import numpy as np
import pandas as p
from PIL import Image
from pathlib import Path
import pickle
from matplotlib import image
import cv2

# dangerous_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "droneSURF/Active_Even_L2/26/"
dangerous_dir = 'Test_AF/deepface/Active_Even_L2/26/'
# for image_file in os.listdir(dangerous_dir):
#     im = image.imread(dangerous_dir / image_file)
#     print(f'Processing for file {image_file}.')
# print(im)
# print(type(im))
# print(len(im))

img = dangerous_dir + "down.jpg"

if os.path.isfile(img) != True:
    raise ValueError("Confirm that ",img," exists")

img = cv2.imread(img)
# f = open(dangerous_dir)
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r" % (cwd))