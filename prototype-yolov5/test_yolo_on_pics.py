import argparse
import os
import math
import random
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
import cv2

import torch

import numpy as np
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def get_files_from_folders_with_ending(folders, ending):
    paths = []
    for folder in folders:
        paths.extend(sorted(
            [
                os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(ending)
            ]
        ))
    return paths

def set_max_img_size(img, max_width):
    img_h, img_w = img.shape[:2]
    r = float(max_width) / img_w
    dim = (max_width, int(img_h * r))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

parser = argparse.ArgumentParser(prog='model-user', description='')
parser.add_argument('-m','--model-path', type=str, default='default', help='')
parser.add_argument('-d','--data-path', type=str, default='renders', help='')
args = parser.parse_args()

root_dir = Path(__file__).resolve().parent
model_path = Path(args.model_path)
testdata_path = Path(args.data_path)
if testdata_path.is_dir():
    print(f'Got dir input')
    testdata_paths = get_files_from_folders_with_ending([testdata_path], (".png", ".jpg"))
else:
    print(f'Got single file input')
    testdata_paths = [testdata_path]

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

for img_path in testdata_paths:
    img = cv2.imread(str(img_path))
    img = set_max_img_size(img, 1000)
    results = model(img)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    cv2.waitKey(0)
cv2.destroyAllWindows()
