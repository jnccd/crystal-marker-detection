import ast
import cv2
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.layers import Flatten, Dense
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
BATCH_SIZE = 32
EPOCHS = 50
NUM_KEYPOINTS = 4 * 2

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/pet-0-pet-test-red-rects'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'
output_folder = create_dir_if_not_exists(root_dir / 'output/mbn')

class KeyPointsLoader(keras.utils.Sequence):
    def __init__(self, input_paths, target_paths, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_paths = input_paths
        self.target_paths = target_paths

    def __len__(self):
        return len(self.input_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_paths = self.input_paths[i : i + self.batch_size]
        batch_target_paths = self.target_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img_to_array(img)
            
        y = np.zeros((self.batch_size,) + (NUM_KEYPOINTS,), dtype="float32")
        for j, path in enumerate(batch_target_paths):
            points = ast.literal_eval(read_textfile(path))
            parr = np.array(points)
            y[j] = np.reshape(parr, (1,1,8))
            
        return x, y
    
train_dataset = KeyPointsLoader(input_paths=get_files_from_folders_with_ending([dataset_train_dir], '.png'),
                                target_paths=get_files_from_folders_with_ending([dataset_train_dir], '.txt'),
                                batch_size=BATCH_SIZE,
                                img_size=(IMG_SIZE, IMG_SIZE))
validation_dataset = KeyPointsLoader(input_paths=get_files_from_folders_with_ending([dataset_val_dir], '.png'),
                                    target_paths=get_files_from_folders_with_ending([dataset_val_dir], '.txt'),
                                    batch_size=BATCH_SIZE,
                                    img_size=(IMG_SIZE, IMG_SIZE))

def get_mbn_model():
    # Load the pre-trained weights of MobileNetV2 and freeze the weights
    backbone = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    backbone.trainable = False

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)
    x = layers.Dropout(0.3)(x)
    x = layers.SeparableConv2D(
        NUM_KEYPOINTS, kernel_size=5, strides=1, activation="relu"
    )(x)
    x = Flatten()(x)
    outputs = Dense(NUM_KEYPOINTS, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="keypoint_detector")

model = get_mbn_model()
model.summary()
model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-4))
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

# Test model
val_img_paths = get_files_from_folders_with_ending([dataset_val_dir], '.png')
for val_img in val_img_paths:
    image = load_img(val_img, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    points = unflatten(preds, 2)
    print(val_img, points)
    
    image = cv2.imread(val_img)
    for i, point in enumerate(points):
        x = int(point[0] * IMG_SIZE)
        y = int(point[1] * IMG_SIZE)
        cv2.circle(image, (x, y), 2, (0, 255 / 4 * i, 0), 2)
    
    cv2.imshow("Model Prediction", image)
    k = cv2.waitKey(0)
    if k == ord('q'):
            break