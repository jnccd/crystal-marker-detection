# https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

import argparse
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

img_size = (320, 320)
boxes_per_image = 4
epochs = 75

root_dir = Path(__file__).resolve().parent
dataseries_t1_dir = root_dir/'..'/'traindata-creator'/'dataseries-320-webcam-images-1312ecab-04e7-4f45-a714-07365d8c0dae'/'images_traindata'
dataseries_v1_dir = root_dir/'..'/'traindata-creator'/'dataseries-320-webcam-images-203d3683-7c91-4429-93b6-be24a28f47bf'/'images_traindata'
dataseries_t2_dir = root_dir/'..'/'traindata-creator'/'dataseries-320-webcam-images-f50ec0b7-f960-400d-91f0-c42a6d44e3d0'/'images_traindata'

data_img_paths = sorted(
    [
        os.path.join(dataseries_t1_dir, fname)
        for fname in os.listdir(dataseries_t1_dir)
        if fname.endswith(".png")
    ]
)
data_img_paths.extend(sorted(
    [
        os.path.join(dataseries_t2_dir, fname)
        for fname in os.listdir(dataseries_t2_dir)
        if fname.endswith(".png")
    ]
))

val_img_paths = sorted(
    [
        os.path.join(dataseries_v1_dir, fname)
        for fname in os.listdir(dataseries_v1_dir)
        if fname.endswith(".png")
    ]
)

def unflatten(list, chunk_size):
    return [list[n:n+chunk_size] for n in range(0, len(list), chunk_size)]

def load_dataset(file_paths, img_size):
    data = []
    targets = []
    filenames = []

    for img_spath in file_paths:
        img_path = Path(img_spath)
        txt_path = img_path.parent / (str(img_path.stem)+'_yolo.txt')
        
        #print((txt_path, img_path))
        boxes = []
        with open(txt_path) as f:
            for row in f.readlines():
                (class_label, x, y, w, h) = row.removesuffix("\n").split(" ")
                boxes.extend((x, y, w, h))
        
        image = load_img(img_path, target_size=(img_size[0], img_size[1]))
        image = img_to_array(image)
        
        data.append(image)
        targets.append(boxes)
        filenames.append(Path(img_path).stem)

    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")
    
    return data, targets, filenames

trainImages, trainTargets, trainFilenames = load_dataset(data_img_paths, img_size)
testImages, testTargets, testFilenames = load_dataset(val_img_paths, img_size)

# Test traindata
for data_img_path, data_target, data_filename in zip(data_img_paths, trainTargets, trainFilenames):
    image = cv2.imread(data_img_path)
    
    rects = unflatten(data_target, 4)
    #print("rects", rects)
    for rect in rects:
        x = int(rect[0] * img_size[0])
        y = int(rect[1] * img_size[1])
        w = int(rect[2] * img_size[0])
        h = int(rect[3] * img_size[1])
        #print(x, y, w, h)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
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
bboxHead = Dense(4*boxes_per_image, activation="sigmoid")(bboxHead)

model = Model(inputs=vgg.input, outputs=bboxHead)

opt = Adam(lr=0.0008)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
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
for val_img in val_img_paths:
    image = load_img(val_img, target_size=(320, 320))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    rects = unflatten(preds, 4)
    print(val_img, rects)
    
    image = cv2.imread(val_img)
    
    for rect in rects:
        x = int(rect[0] * img_size[0])
        y = int(rect[1] * img_size[1])
        w = int(rect[2] * img_size[0])
        h = int(rect[3] * img_size[1])
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Model Prediction", image)
    k = cv2.waitKey(0)
    if k == ord('q'):
            break