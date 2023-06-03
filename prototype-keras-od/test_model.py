import argparse
import os
import sys
import time
import cv2
from pathlib import Path

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import cv2
import os

def unflatten(list, chunk_size):
    return [list[n:n+chunk_size] for n in range(0, len(list), chunk_size)]

img_size = (320, 320)
boxes_per_image = 4

root_dir = Path(__file__).resolve().parent
dataseries_v1_dir = root_dir/'..'/'traindata-creator'/'dataseries-320-webcam-images-f50ec0b7-f960-400d-91f0-c42a6d44e3d0'/'images_traindata'

model = load_model(root_dir / 'model.h5')

val_img_paths = sorted(
    [
        os.path.join(dataseries_v1_dir, fname)
        for fname in os.listdir(dataseries_v1_dir)
        if fname.endswith(".png")
    ]
)
for val_img in val_img_paths:
    image = load_img(val_img, target_size=(320, 320))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    print(preds)
    rects = unflatten(preds, 4)
    print(rects)
    
    image = cv2.imread(val_img)
    
    for rect in rects:
        x = int(rect[0] * img_size[0])
        y = int(rect[1] * img_size[1])
        w = int(rect[2] * img_size[0])
        h = int(rect[3] * img_size[1])
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Output", image)
    cv2.waitKey(0)