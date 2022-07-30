import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from os.path import expanduser
from pathlib import Path
import pickle
import os


im = Path(os.environ["ICHOR_INPUT_DATASET"]) / "droneSURF/Active_Even_L1/1/"
im = im.parts
im = '/'.join(im)
im = im + '/'


im1 = im + 'down.jpg'
im2 = im + 'front.jpg'

from deepface import DeepFace

model_name = "Dlib"
distance_metric = "euclidean_l2"
detector_backend = 'opencv'


h = DeepFace.verify(img1_path=im1, img2_path=im1, model_name = model_name, distance_metric = distance_metric, enforce_detection = False, detector_backend = detector_backend)
