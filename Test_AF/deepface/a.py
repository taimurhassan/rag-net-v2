import os
import tqdm
import numpy as np
import pandas as p
from PIL import Image
from pathlib import Path
import pickle
from matplotlib import image


dangerous_dir = Path(os.environ["ICHOR_INPUT_DATASET"]) / "COMPASS-XP" / "Dangerous"
for image_file in os.listdir(dangerous_dir):
    im = image.imread(dangerous_dir / image_file)
    print(f'Processing for file {image_file}.')
print(im)