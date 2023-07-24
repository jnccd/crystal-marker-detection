import argparse
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from pathlib import Path

parser = argparse.ArgumentParser(prog='tes-stuff', description='test stuff.')
parser.add_argument('-r','--run', type=str, default='exp6', help='Yolov5 run foldername.')
args = parser.parse_args()

run = args.run

root_dir = Path(__file__).resolve().parent
network_file = str(root_dir / f'../repos/yolov5/runs/train/{run}/weights/best.pt')

print("network_file",network_file)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=network_file, force_reload=True)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()