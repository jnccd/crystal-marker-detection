import argparse
import os
import math
import random
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img, array_to_img

parser = argparse.ArgumentParser(prog='model-user', description='')
parser.add_argument('-m','--model-path', type=str, default='default', help='')
parser.add_argument('-d','--data-path', type=str, default='renders', help='')
args = parser.parse_args()

model = keras.models.load_model(Path(args.model_path))
model_pred = model.predict(load_img(Path(args.data_path)))

out_img = ImageOps.autocontrast(array_to_img(model_pred))
out_img.save('model_pred.png')