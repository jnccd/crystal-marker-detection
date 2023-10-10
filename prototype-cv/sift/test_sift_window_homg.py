import random
import sys
import cv2
import numpy as np
from pathlib import Path

from utils import *

def apply_homography(point2D_list, h, convert_to_int = True):
    hps = [h @ (p[0], p[1], 1) for p in point2D_list] 
    ps = [(p[0] / p[2], p[1] / p[2]) for p in hps]
    
    if convert_to_int:
        return [(int(p[0]), int(p[1])) for p in ps]
    else:
        return ps

root_dir = Path(__file__).resolve().parent
border_marker_img_path = root_dir/'sift-base'/'border-marker.png'
in_img_marker_img_path = root_dir/'sift-base'/'in-img-marker.png'
test_img_path = Path("N:\Downloads\Archives\FabioBilder\\the_good_pics_for_sift\DSC_3741.JPG")

marker_img = cv2.imread(str(in_img_marker_img_path))
test_img = cv2.imread(str(test_img_path))

marker_img = cv2.resize(marker_img, dsize=(200,200), interpolation=cv2.INTER_NEAREST)
test_img = keep_image_size_in_check(test_img)
img_h, img_w = test_img.shape[:2]
mimg_h, mimg_w = marker_img.shape[:2]

# Threshold
img_t = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
img_t = cv2.adaptiveThreshold(img_t,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,101,2)
img_t = 255-img_t

# cv2.imshow('img',img_t)
# cv2.waitKey(0)

orb = cv2.ORB_create()
sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

match_points = []
window_size_div2 = 250
breakk = False
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
        print('readable_god_matches', readable_good_matches)
        
        normalized_good_matches = [(y[0] - x[0], y[1] - x[1]) for x, y in readable_good_matches]
        print('normalized_good_matches', normalized_good_matches)
        
        draw_img = cv2.drawMatches(marker_img, kp1, img_window, kp2, good_matches, img_window, flags=2)
        
        for m in range(len(readable_good_matches)-1):
            cutted_rgms = readable_good_matches.copy()
            cutted_rgms.pop(m)
            
            for m2 in range(len(cutted_rgms)-1):
                cutted_rgms2 = cutted_rgms.copy()
                cutted_rgms2.pop(m2)
                
                # Find Homography
                src_rect  = np.array([[x[0], x[1], 1] for x,y in cutted_rgms2])
                dest_rect = np.array([[y[0], y[1], 1] for x,y in cutted_rgms2])
                h, status = cv2.findHomography(src_rect, dest_rect)
                
                # Mark found rectangles in inner_rect
                mapped_corners = apply_homography([(0,0), (0,mimg_h), (mimg_w,mimg_h), (mimg_w,0)], h)
                for i in range(0, len(mapped_corners)):
                    col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    if i % 4 == 3:
                        cv2.line(draw_img, mapped_corners[i], mapped_corners[i-3], col, 2)
                    else:
                        cv2.line(draw_img, mapped_corners[i], mapped_corners[i+1], col, 2)
        
        cv2.imshow('img', draw_img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            breakk = True
            break
    if breakk:
        break
        
print(match_points)
img_draw = np.copy(test_img)
for p in match_points:
    cv2.circle(img_draw, (int(p[0]),int(p[1])), 5, (0,0,255))
#cv2.imwrite(str(root_dir / f'{test_img_path.stem}_sift_matches.png'), img_draw)