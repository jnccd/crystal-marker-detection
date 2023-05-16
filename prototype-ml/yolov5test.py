import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = 'https://ultralytics.com/images/zidane.jpg'

results = model(img)
results.show()
