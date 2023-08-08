import math
import sys
import cv2
import numpy as np
from pathlib import Path

img_img_filename = 'DSC_3741.JPG'
root_dir = Path(__file__).resolve().parent
test_img_path = root_dir / img_img_filename
test_img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)

# Gen img pyramid
pyr_imgs = [test_img]
while pyr_imgs[-1].shape[0] > 1000:
    pyr_h, pyr_w = pyr_imgs[-1].shape[:2]
    pyr_imgs.append(cv2.pyrDown(pyr_imgs[-1], dstsize=(int(pyr_w/2), int(pyr_h/2))))
#print(len(pyr_imgs))

img = pyr_imgs[-1]

block_size = 150#int(img_w/600*50)
block_size = block_size if block_size % 2 == 1 else block_size + 1
img_t = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,block_size,3)
img_t = (255-img_t)
cv2.imshow('img',img_t)
cv2.waitKey(0)

def get_opencv_aruco_detector(dict):
    dictionary = cv2.aruco.getPredefinedDictionary(dict)
    parameters =  cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)

def aruco_transform_and_display(corners, ids, rejected, image):
    
    centers = []
    out_corners = []
    
    if len(corners) > 0:
        ids = ids.flatten()
        
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
			
            out_corners.append([topLeft, topRight, bottomRight, bottomLeft])
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            cv2.rectangle(image, topLeft, (topLeft[0]+1, topLeft[1]+1), (0, 0, 255), 5)
			
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            centers.append((cX, cY))
			
            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
			
    return image, centers, out_corners

detector = get_opencv_aruco_detector(cv2.aruco.DICT_6X6_50)
corners, ids, rejected = detector.detectMarkers(img_t)
marked_img, centers, cornerss = aruco_transform_and_display(corners, ids, rejected, img_t.copy())
cv2.imshow('img',marked_img)
cv2.waitKey(0)