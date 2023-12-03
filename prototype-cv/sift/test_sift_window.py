import sys
import cv2
import numpy as np
from pathlib import Path

from utils import *

root_dir = Path(__file__).resolve().parent
border_marker_img_path = root_dir/'sift-base'/'border-marker.png'
in_img_marker_img_path = root_dir/'sift-base'/'in-img-marker.png'
test_img_path = Path("N:\Downloads\Archives\FabioBilder\\the_good_pics_for_sift\DSC_3741.JPG")

marker_img = cv2.imread(str(in_img_marker_img_path))
test_img = cv2.imread(str(test_img_path))
print(test_img.shape)

marker_img = cv2.resize(marker_img, dsize=(200,200), interpolation=cv2.INTER_NEAREST)
#marker_img = cv2.flip(marker_img, 0)
test_img = keep_image_size_in_check(test_img)
img_h, img_w = test_img.shape[:2]

# Threshold
img_t = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
img_t = cv2.adaptiveThreshold(img_t,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,101,2)
img_t = 255-img_t
cv2.imwrite(str(root_dir / f'{test_img_path.stem}_thresh_sift_matches.png'), img_t)

# cv2.imshow('img',img_t)
# cv2.waitKey(0)

orb = cv2.ORB_create()
sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

match_points = []
window_size_div2 = 250
for x in range(window_size_div2, img_w-window_size_div2, 150):
    for y in range(window_size_div2, img_h-window_size_div2, 150):
        y_min = y-window_size_div2
        y_max = y+window_size_div2
        x_min = x-window_size_div2
        x_max = x+window_size_div2
        
        img_window = img_t[y_min:y_max, x_min:x_max]

        kp1, des1 = sift.detectAndCompute(marker_img,None)
        kp2, des2 = sift.detectAndCompute(img_window,None)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x: x.distance)
        good_matches = matches[:int(len(matches)/2)]
        
        #print('kpls',len(kp1), 'kp2s',len(kp2), 'matches',len(matches), 'good_matches',len(good_matches))
        readable_good_matches = [(kp1[dmatch.queryIdx].pt, kp2[dmatch.trainIdx].pt) for dmatch in good_matches]
        #print('readable_god_matches', readable_good_matches)
        
        normalized_good_matches = [(y[0] - x[0], y[1] - x[1]) for x, y in readable_good_matches]
        #print('normalized_good_matches', normalized_good_matches)
        
        stds = (np.std([x[0] for x in normalized_good_matches]), np.std([x[1] for x in normalized_good_matches]))
        print(f'{x_min, y_min}, stds', stds)
        
        if all([x < 90 for x in stds]):
            match_points.append((np.mean([y[0] for x, y in readable_good_matches]) + x_min, 
                                 np.mean([y[1] for x, y in readable_good_matches]) + y_min
                                 ))
        
        matches_img = cv2.drawMatches(marker_img, kp1, img_window, kp2, good_matches, img_window, flags=2)
        cv2.imshow('img',matches_img)
        cv2.waitKey(0)
        
print(match_points)
img_draw = np.copy(test_img)
for p in match_points:
    cv2.circle(img_draw, (int(p[0]),int(p[1])), 15, (0,0,255), 5)
cv2.imwrite(str(root_dir / f'{test_img_path.stem}_sift_matches.png'), img_draw)