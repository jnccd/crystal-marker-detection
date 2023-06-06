import cv2
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent
border_marker_img_path = root_dir/'sift-base'/'border-marker-scaled.png'
in_img_marker_img_path = root_dir/'sift-base'/'in-img-marker-scaled-arranged.png'
test_img_path = Path("N:\Downloads\Archives\FabioBilder\\the_good_pics_for_sift\DSC_3741.JPG")

img1 = cv2.imread(str(in_img_marker_img_path))
img2 = cv2.imread(str(test_img_path))

size_mult = 2

img2 = cv2.resize(img2, dsize=(600*size_mult, 400*size_mult), interpolation=cv2.INTER_CUBIC)

cv2.imshow('in img',img2)

background_col = np.array([151, 164, 186])
foreground_col = np.array([150, 139, 127])

imgg = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#imgr = cv2.inRange(img2, foreground_col, background_col)
imgg = cv2.adaptiveThreshold(imgg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,50*size_mult+1,2)
imgg = 255-imgg
#imgg = cv2.GaussianBlur(imgg,(3,3),0)
final_img = imgg

cv2.imshow('Thresholded img',final_img)
cv2.waitKey(0)

orb = cv2.ORB_create()
sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

kpl, desl = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(final_img,None)

matches = bf.match(desl,des2)
matches = sorted(matches, key = lambda x:x.distance)

print(len(matches))

img3 = cv2.drawMatches(img1, kpl, final_img, kp2, matches[:20], final_img, flags=2)

cv2.imshow('SIFT',img3)
cv2.waitKey(0)
