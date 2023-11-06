import math
from pathlib import Path
import random
import cv2
import numpy as np

def set_img_width(img, max_width):
    img_h, img_w = img.shape[:2]
    resize_factor = float(max_width) / img_w
    target_size = (max_width, int(img_h * resize_factor))
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def set_img_height(img, max_height):
    img_h, img_w = img.shape[:2]
    resize_factor = float(max_height) / img_h
    target_size = (int(img_w * resize_factor), max_height)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def keep_image_size_in_check(img, max_img_width=1920, max_img_height=1080):
    img_h, img_w = img.shape[:2]
    if img_w > max_img_width:
        img = set_img_width(img, max_img_width)
    if img_h > max_img_height:
        img = set_img_height(img, max_img_height)
    return img

img_img_filename = 'DSC_3741.JPG'
root_dir = Path(__file__).resolve().parent
test_img_path = root_dir / img_img_filename
test_img = cv2.imread(str(test_img_path))#, cv2.IMREAD_GRAYSCALE)
test_img = keep_image_size_in_check(test_img)
gray_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# Adaptive thresh
block_size = 90#int(img_w/600*50)
block_size = block_size if block_size % 2 == 1 else block_size + 1
thresh_test_img = cv2.adaptiveThreshold(gray_test_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,block_size,3)

# Get corners
np_test_img = np.float32(thresh_test_img)
dst = cv2.cornerHarris(np_test_img,11,5,0.04)
ret, dst = cv2.threshold(dst,0.2*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners: np.ndarray = cv2.cornerSubPix(np_test_img,np.float32(centroids),(5,5),(-1,-1),criteria)

print(len(corners))

def distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def diff(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])

# # Group and filter corners
# for a in range(len(corners)):
#     for b in [x for x in range(len(corners)) if x != a]:
#         for c in [x for x in range(len(corners)) if x != a and x != b]:
#             print(a,b,c)
#             print(corners[a], corners[b], corners[c])
            
#             # Distances similarity check
#             distance_rato = distance(corners[a],corners[b]) / distance(corners[b],corners[c])
#             if distance_rato > 1:
#                 distance_rato = 1 / distance_rato
#             if distance_rato < 0.7:
#                 continue
            
#             # Angle check
#             ba = diff(corners[a],corners[b])
#             bc = diff(corners[c],corners[b])
#             angle = math.atan2(bc[1], bc[0]) - math.atan2(ba[1], ba[0])
#             angle = angle + 2*math.pi if angle < 0 else angle
#             if angle < math.pi - 0.1 or angle > math.pi + 0.1:
#                 continue
            
# Display
#test_img[dst>0.1*dst.max()]=[0,0,255]
draw_img = cv2.cvtColor(thresh_test_img, cv2.COLOR_GRAY2BGR)
draw_img = 255 - draw_img
for corner in corners:
    cv2.circle(draw_img, (int(corner[0]), int(corner[1])), 2, (0,0,255), 3)
cv2.imshow('image', draw_img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    pass