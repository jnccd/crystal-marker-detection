import argparse
import json
import os
import math
import random
from pathlib import Path
import shutil
import sys
import cv2
import re
import numpy as np

from utility import *

# Parse
parser = argparse.ArgumentParser(prog='', description='Regenerate evaldata and reanalyze outputs, in case there was a bug in the prod server script.')
parser.add_argument('-r','--run-path', type=str, help='.')
parser.add_argument('-rt','--run-type', type=str, help='.')
args = parser.parse_args()

print(Path(args.run_path))
run_dir, train_def_path = [(x.parent.parent, x) for x in Path(args.run_path).glob('**/training-def.json')][0]
best_weights_path = run_dir / 'weights' / 'best.pt'
print(f'Running webcam test on the following run: {run_dir}')

yolov5_pattern = re.compile('yolov5(.?)$')
yolov8_pattern = re.compile('yolov8(.?)$|yolov5(.?)u$') 

# Read train def
train_def_json = json.loads(read_textfile(train_def_path).replace("    ", "").replace("\n", ""))
train_def_model = train_def_json['model']

if yolov5_pattern.match(train_def_model):
    import torch
    
    # Load model
    network_file = str(best_weights_path)
    print("network_file",network_file)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=network_file, force_reload=True)

    # Main loop
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        results = model(frame)
        
        cv2.imshow('YOLO', np.squeeze(results.render()))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)
elif yolov8_pattern.match(train_def_model):
    from ultralytics import YOLO
    
    # Load model
    network_file = str(best_weights_path)
    print("network_file",network_file)
    model = YOLO(network_file)

    # Main loop
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        result = model(frame)[0]
        
        cv2.imshow('YOLO', np.squeeze(result.plot()))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)
else:
    print('What model is that? ' + train_def_model)
    sys.exit(1)