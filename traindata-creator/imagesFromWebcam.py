from matplotlib import pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import os
import time

root_dir = Path(__file__).resolve().parent
images_path = root_dir / 'images'
if not os.path.exists(images_path):
    os.makedirs(images_path)

cap = cv2.VideoCapture(0)

i=0
while True:
    ret, frame = cap.read()
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord(' '):
        img_path = str(images_path / f"img{i}.png")
        cv2.imwrite(img_path, frame)
        print("Storing image at", img_path)
        i += 1

    cv2.imshow('Image Collection', frame)
    
cap.release()
cv2.destroyAllWindows()