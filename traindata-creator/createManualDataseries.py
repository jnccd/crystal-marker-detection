import argparse
import ast
import os
import random
import sys
import time
from typing import Literal
import cv2
from pathlib import Path
from cv2 import Mat
import numpy as np
from shapely import LineString, Point, Polygon

from utils import *

# globals
window_name = "Manual Poly Dataseries Creator"
cur_poly_points = []
cur_polys = []

def main():
    global window_name, cur_m_pos, cur_poly_points
    
    parser = argparse.ArgumentParser(prog='manual-dataseries-creator', description='Creates dataseries from manually marked traindata.')
    parser.add_argument('-if','--input-folder', type=str, help='The path to the folder containing an image series.')
    parser.add_argument('-pmw','--preresize-max-width', type=int, default=1920, help='The maximum width for input images.')
    parser.add_argument('-pmh','--preresize-max-height', type=int, default=1080, help='The maximum height for input images.')
    args = parser.parse_args()
    
    # Prepare paths
    root_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_folder)
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.lower().endswith((".png", ".jpg"))
        ]
    )
    dataseries_dir = root_dir / 'dataseries' / f'man-{input_dir.name}'
    if not os.path.exists(dataseries_dir):
        os.makedirs(dataseries_dir)
    
    # Load imgs
    in_imgs = [keep_image_size_in_check(cv2.imread(str(p)), args.preresize_max_width, args.preresize_max_height) for p in input_img_paths]
    imgs_polys = [[] for x in range(len(in_imgs))]
    imgs_index = 0
    #img_h, img_w = img.shape[:2]

    # Load window and hook events
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouseEvent)
    
    while True:
        # Control logic at 60 FPS
        k = cv2.waitKey(16) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('d') or k == ord(' '):
            imgs_polys[imgs_index] = cur_polys
            cur_poly_points = []
            
            imgs_index += 1
            imgs_index = imgs_index % len(in_imgs)
            
            cur_polys = imgs_polys[imgs_index]
        elif k == ord('a'):
            imgs_polys[imgs_index] = cur_polys
            
            cur_poly_points = []
            
            imgs_index -= 1
            imgs_index = imgs_index % len(in_imgs)
            
            cur_polys = imgs_polys[imgs_index]
        elif k == ord('c'):
            print("c")
        elif k == ord('z'):
            print("z")
        elif k == ord('f'):
            print("f")
        
        # Draw
        display_img = in_imgs[imgs_index].copy()
        for poly in cur_polys:
            pts = np.array([poly], np.int32).reshape((-1,1,2))
            cv2.polylines(display_img,[pts],True,(0,255,0))
        # Draw cur poly
        pts = np.array([cur_poly_points], np.int32).reshape((-1,1,2))
        cv2.polylines(display_img,[pts],False,(255,0,0))
        cv2.imshow(window_name, display_img)
    
    cv2.destroyAllWindows()

def mouseEvent(action, x, y, flags, *userdata):
    global window_name, cur_m_pos, cur_poly_points
    
    cur_m_pos = (x,y)
    if action == cv2.EVENT_LBUTTONDOWN:
        print('M1 down')
    elif action == cv2.EVENT_LBUTTONUP:
        if len(cur_poly_points) > 0:
            sx, sy = cur_poly_points[0]
            dx = sx - x
            dy = sy - y
            if dx*dx + dy*dy < 5*5:
                cur_polys.append(cur_poly_points)
                cur_poly_points = []
            else:
                cur_poly_points.append(cur_m_pos)
        else:
            cur_poly_points.append(cur_m_pos)

if __name__ == '__main__':
    main()