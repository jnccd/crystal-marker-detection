# https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

import argparse
import ast
import os
import sys
import time
import cv2
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from utils import *

img_size = (160, 160)
epochs = 75

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/pet-0-pet-test-red-rects'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'

def load_dataseries(dataseries_path, img_size):
    data = []
    targets = []
    filenames = []
    
    img_paths = get_files_from_folders_with_ending([dataseries_path], '.png')
    vert_paths = get_files_from_folders_with_ending([dataseries_path], '.txt')

    for img_path, vert_path in zip(img_paths, vert_paths):
        image = load_img(img_path, target_size=(img_size[0], img_size[1]))
        image = img_to_array(image)
        
        points = ast.literal_eval(read_textfile(vert_path))
        
        data.append(image)
        targets.append(flatten([(x / img_size[0], y / img_size[1]) for (x, y) in points]))
        filenames.append(Path(img_path).stem)

    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")
    
    return data, targets, filenames

train_np_images, train_targets, train_filenames = load_dataseries(dataset_train_dir, img_size)
val_np_images, val_targets, val_filenames = load_dataseries(dataset_val_dir, img_size)

# Test traindata
train_img_paths = get_files_from_folders_with_ending([dataset_train_dir], '.png')
for data_img_path, data_target, data_filename in zip(train_img_paths, train_targets, train_filenames):
    image = cv2.imread(data_img_path)
    
    points = unflatten(data_target, 2)
    for point in points:
        x = int(point[0] * img_size[0])
        y = int(point[1] * img_size[1])
        cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
    
    print(f"Traindata {data_filename}")
    cv2.imshow("Traindata", image)
    k = cv2.waitKey(0)
    if k == ord('q'):
            break

# Model
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(img_size[0], img_size[1], 3)))
vgg.trainable = False

flatten = vgg.output
#print(vgg.output)
flatten = Flatten()(flatten)

bboxHead = Dense(256, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(8, activation="sigmoid")(bboxHead)

model = Model(inputs=vgg.input, outputs=bboxHead)

opt = Adam(lr=0.0008)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

print('fitting...')
H = model.fit(
	train_np_images, train_targets,
	validation_data=(val_np_images, val_targets),
	batch_size=8,
	epochs=epochs,
	verbose=1)

print("[INFO] saving object detector model...")
model.save(root_dir / 'model.h5', save_format="h5")

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(root_dir / 'plot.png')

# Test model
val_img_paths = get_files_from_folders_with_ending([dataset_val_dir], '.png')
for val_img in val_img_paths:
    image = load_img(val_img, target_size=img_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    points = unflatten(preds, 2)
    print(val_img, points)
    
    image = cv2.imread(val_img)
    
    points = unflatten(preds, 2)
    for point in points:
        x = int(point[0] * img_size[0])
        y = int(point[1] * img_size[1])
        cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
    
    cv2.imshow("Model Prediction", image)
    k = cv2.waitKey(0)
    if k == ord('q'):
            break