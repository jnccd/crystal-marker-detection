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
window_name = "Manual vert Dataseries Creator"
cur_m_pos = (0, 0)
cur_vert_points = []
cur_verts = []
output_img_paths = []
output_vert_paths = []
in_imgs = []
imgs_verts = []

def main():
    global window_name, cur_m_pos, cur_vert_points, cur_verts, output_img_paths, output_vert_paths, in_imgs, imgs_verts
    
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
            Path(os.path.join(input_dir, fname))
            for fname in os.listdir(input_dir)
            if fname.lower().endswith((".png", ".jpg"))
        ]
    )
    dataseries_dir = root_dir / 'dataseries' / f'man-{input_dir.name}'
    if not os.path.exists(dataseries_dir):
        os.makedirs(dataseries_dir)
    output_img_paths = [dataseries_dir / f'{in_img_p.stem}_in.png' for in_img_p in input_img_paths]
    output_vert_paths = [dataseries_dir / f'{in_img_p.stem}_vertices.txt' for in_img_p in input_img_paths]
    
    # Load imgs
    in_imgs = [keep_image_size_in_check(cv2.imread(str(p)), args.preresize_max_width, args.preresize_max_height) for p in input_img_paths]
    imgs_verts = [ast.literal_eval(read_textfile(output_vert_paths[i])) if output_vert_paths[i].is_file() else [] for i in range(len(in_imgs))]
    imgs_index = 0
    cur_verts = imgs_verts[imgs_index]
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
            finalize_data_tuple(imgs_index)
            
            imgs_index += 1
            imgs_index = imgs_index % len(in_imgs)
            
            cur_verts = imgs_verts[imgs_index]
        elif k == ord('a'):
            finalize_data_tuple(imgs_index)
            
            imgs_index -= 1
            imgs_index = imgs_index % len(in_imgs)
            
            cur_verts = imgs_verts[imgs_index]
        elif k == ord('c'):
            cur_verts = []
        elif k == ord('z'):
            cur_verts.pop()
        elif k == ord('f'):
            cur_verts.append(cur_vert_points)
            cur_vert_points = []
        
        # Draw
        display_img = in_imgs[imgs_index].copy()
        for vert in cur_verts:
            pts = np.array([vert], np.int32).reshape((-1,1,2))
            cv2.polylines(display_img,[pts],True,(0,255,0))
        # Draw cur vert
        pts = np.array([cur_vert_points], np.int32).reshape((-1,1,2))
        cv2.polylines(display_img,[pts],False,(255,0,0))
        cv2.imshow(window_name, display_img)
    
    cv2.destroyAllWindows()
    
def finalize_data_tuple(i):
    global window_name, cur_m_pos, cur_vert_points, cur_verts, output_img_paths, output_vert_paths, in_imgs, imgs_verts
    
    imgs_verts[i] = cur_verts
    cur_vert_points = []
    
    write_textfile(str(cur_verts), output_vert_paths[i])
    cv2.imwrite(str(output_img_paths[i]), in_imgs[i])

def mouseEvent(action, x, y, flags, *userdata):
    global window_name, cur_m_pos, cur_vert_points, cur_verts, output_img_paths, output_vert_paths, in_imgs, imgs_verts
    
    cur_m_pos = (x,y)
    if action == cv2.EVENT_LBUTTONDOWN:
        print('Vertex created')
    elif action == cv2.EVENT_LBUTTONUP:
        if len(cur_vert_points) > 0:
            sx, sy = cur_vert_points[0]
            dx = sx - x
            dy = sy - y
            if dx*dx + dy*dy < 5*5:
                cur_verts.append(cur_vert_points)
                cur_vert_points = []
            else:
                cur_vert_points.append(cur_m_pos)
        else:
            cur_vert_points.append(cur_m_pos)

if __name__ == '__main__':
    main()