from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import load_img
import tensorflow as tf

from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import imgaug.augmenters as iaa

from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import os

from utils import *

IMG_SIZE = 160
BATCH_SIZE = 64
EPOCHS = 50
NUM_KEYPOINTS = 4 * 2

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/pet-0-pet-test-red-rects'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'
output_folder = create_dir_if_not_exists(root_dir / 'output/mbn')

class KeyPointsDataset(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_paths, target_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_paths = input_paths
        self.target_paths = target_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_paths = self.input_paths[i : i + self.batch_size]
        batch_target_paths = self.target_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
            
        y = np.zeros((self.batch_size,) + (NUM_KEYPOINTS,), dtype="float32")
        for j, path in enumerate(batch_target_paths):
            y[j] = load_img(path, target_size=self.img_size, color_mode="grayscale")
            
        return x, y