from matplotlib import pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import uuid
import os
import time

root_dir = Path(__file__).resolve().parent
images_path = root_dir / 'images'
if not os.path.exists(images_path):
    os.makedirs(images_path)

number_imgs = 30
label="marker"

cap = cv2.VideoCapture(0)
# Loop through labels
print('Collecting images for {}'.format(label))
time.sleep(5)

# Loop through image range
for img_num in range(number_imgs):
    print('Collecting images for {}, image number {}'.format(label, img_num))
        
    # Webcam feed
    ret, frame = cap.read()
        
    # Naming out image path
    imgname = str(images_path / (label+'.'+str(uuid.uuid1())+'.jpg'))
        
    # Writes out image to file 
    cv2.imwrite(imgname, frame)
    print("Storing image at", imgname)
        
    # Render to the screen
    cv2.imshow('Image Collection', frame)
        
    # 2 second delay between captures
    time.sleep(2)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()