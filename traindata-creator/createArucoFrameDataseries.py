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
window_name = "Window"
top_left_corner=[]
bottom_right_corner=[]
new_top_left = None
cur_m_pos = None
g_img_resize_factor = 1

def main():
    global window_name, top_left_corner, bottom_right_corner, new_top_left, cur_m_pos, g_img_resize_factor
    
    parser = argparse.ArgumentParser(prog='aruco-frame-dataseries-creator', description='Automatically creates traindata from objects on planes within 3D image space that are spanned by 4 aruco markers.')
    parser.add_argument('-if','--input-folder', type=str, help='The path to the folder containing an image series.')
    parser.add_argument('-s','--size', type=int, default=0, help='The width and height of the traindata images.')
    parser.add_argument('-ng','--no-gui', action='store_true', help='Builds traindata immediately based on cached label data.')
    parser.add_argument('-pmw','--preresize-max-width', type=int, default=1920, help='The maximum width for input images.')
    parser.add_argument('-pmh','--preresize-max-height', type=int, default=1080, help='The maximum height for input images.')
    parser.add_argument('-lirf','--legacy-rect-finding', action='store_true', default=False, help='Use old rect finding based on aruco marker pos relative to the center point.')
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
    dataseries_dir = root_dir / 'dataseries' / f'af-{input_dir.name}'
    marked_dir = dataseries_dir / 'images_marked'
    if not os.path.exists(marked_dir):
        os.makedirs(marked_dir)
    train_dir = dataseries_dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    # Load first image and preprocess
    img = cv2.imread(input_img_paths[0])
    if args.preresize_max_width > 0 or args.preresize_max_height > 0:
        # Resize image to displayable sizes if necessary (neural network inputs are so small compared to this that it should not matter and is more convenient for displaying and for performance)
        l_img_h, l_img_w = img.shape[:2]
        img = keep_image_size_in_check(img, args.preresize_max_width, args.preresize_max_height) 
        img_h, img_w = img.shape[:2]
        g_img_resize_factor = img_w / l_img_w
    else:
        img_h, img_w = img.shape[:2]
    print(img_w, img_h, g_img_resize_factor)
    detector = get_opencv_aruco_detector(cv2.aruco.DICT_6X6_50)
    oh, hi, marked_img, in_between_rect = find_homography_from_aruco(img, detector, img_w, img_h, args.legacy_rect_finding)
    if hi is None:
        cv2.imshow(window_name, marked_img)
        print("Didn't find the aruco frame in base img :/")
        cv2.waitKey(0)
        return
    warped_img = cv2.warpPerspective(img, hi, (img_w, img_h))
    
    # Load markers from last execution
    persistence_file_path = input_dir / ".traindata_markings"
    if os.path.exists(persistence_file_path) and os.path.isfile(persistence_file_path):
        with open(persistence_file_path) as f:
            persistence_str = f.read()
        persistence_dict = ast.literal_eval(persistence_str)
        top_left_corner = persistence_dict['top_left_corner']
        bottom_right_corner = persistence_dict['bottom_right_corner']
        
    if args.no_gui:
        print("Building...")
        build_traindata(
            input_img_paths, 
            detector, 
            img_w, img_h, 
            marked_dir, 
            train_dir, 
            args.size, 
            use_legacy_rect_finding=args.legacy_rect_finding, 
            show_gui=False,
            )
        return
    
    # Load window and hook events
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouseEvent)
    
    while True:
        # Control logic at 60 FPS
        k = cv2.waitKey(16) & 0xFF
        if k == ord('q'):
            # Save marked rects
            with open(persistence_file_path, "w") as text_file:
                persistence_dict = { 'top_left_corner': top_left_corner, 'bottom_right_corner': bottom_right_corner }
                text_file.write(str(persistence_dict))
            break
        elif k == ord('c'):
            top_left_corner=[]
            bottom_right_corner=[]
            new_top_left = None
        elif k == ord('z'):
            top_left_corner.pop()
            bottom_right_corner.pop()
        elif k == ord(' '):
            print("Building...")
            build_traindata(
                input_img_paths, 
                detector, 
                img_w, img_h, 
                marked_dir, 
                train_dir, 
                args.size, 
                use_legacy_rect_finding=args.legacy_rect_finding,
                )

        # Draw
        display_img = warped_img.copy()
        for i in range(0, len(bottom_right_corner)):
            cv2.rectangle(display_img, top_left_corner[i], bottom_right_corner[i], (0,255,0), 1, 8)
        if new_top_left is not None:
            cv2.rectangle(display_img, new_top_left, cur_m_pos, (0,255,0), 1, 8)
        cv2.imshow(window_name, display_img)
    
    cv2.destroyAllWindows()

def mouseEvent(action, x, y, flags, *userdata):
    global window_name, top_left_corner, bottom_right_corner, new_top_left, cur_m_pos
    
    cur_m_pos = (x,y)
    if action == cv2.EVENT_LBUTTONDOWN:
        new_top_left = (x,y)
    elif action == cv2.EVENT_LBUTTONUP:
        top_left_corner.append(new_top_left)
        bottom_right_corner.append((x,y))
        new_top_left = None
        
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

def find_inner_rect(cornerss, ccx, ccy):
    in_between_rect = [None, None, None, None]
    for corners in cornerss:
        
        # Get closest corner to middle
        min_i = sys.maxsize
        min_v = sys.maxsize
        for i in range(len(corners)):
            v = corners[i]
            cv = (v[0]-ccx)*(v[0]-ccx) + (v[1]-ccy)*(v[1]-ccy)
            if cv <= min_v:
                min_i = i
                min_v = cv
                
        # The corner closest to the middle is the min_i'th vertex of the inner rect
        min_vert = corners[min_i]
        while in_between_rect[min_i] is not None:
            min_i += 1
            min_i = min_i % 4
        in_between_rect[min_i] = (int(min_vert[0]), int(min_vert[1]))
        
    return in_between_rect
    
def find_inner_rect_legacy(cornerss, ccx, ccy):
    in_between_rect = [None, None, None, None]
    for corners in cornerss:
        
        # Get closest corner to middle
        min_i = sys.maxsize
        min_v = sys.maxsize
        for i in range(len(corners)):
            v = corners[i]
            cv = (v[0]-ccx)*(v[0]-ccx) + (v[1]-ccy)*(v[1]-ccy)
            if cv <= min_v:
                min_i = i
                min_v = cv
                
        # Sort markers into in_between_rect based on pos relative to center's center
        min_vert = corners[min_i]
        ibr_index = -1
        if min_vert[0] < ccx:
            if min_vert[1] < ccy:
                ibr_index = 1
            else:
                ibr_index = 0
        else:
            if min_vert[1] < ccy:
                ibr_index = 2
            else:
                ibr_index = 3
        
        while in_between_rect[ibr_index] is not None:
            ibr_index += 1
            ibr_index = ibr_index % 4
        in_between_rect[ibr_index] = (int(min_vert[0]), int(min_vert[1]))
        
    return in_between_rect
    
def get_opencv_aruco_detector(dict):
    dictionary = cv2.aruco.getPredefinedDictionary(dict)
    parameters =  cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)
    
def find_homography_from_aruco(img, detector, width, height, use_legacy_rect_finding=False):
    corners, ids, rejected = detector.detectMarkers(img)
    marked_img, centers, cornerss = aruco_transform_and_display(corners, ids, rejected, img.copy())
    
    # Found paper
    if len(corners) == 4:
        # Find center
        ccx = int(np.mean([c[0] for c in centers]))
        ccy = int(np.mean([c[1] for c in centers]))
        cv2.rectangle(marked_img, (ccx, ccy), (ccx+1, ccy+1), (0, 0, 255), 5)
        
        # Find inner rectangle
        if (use_legacy_rect_finding):
            in_between_rect = find_inner_rect_legacy(cornerss, ccx, ccy)
        else:
            in_between_rect = find_inner_rect(cornerss, ccx, ccy)
        for i in range(0,4):
            j = (i+1)%4
            cv2.line(marked_img, in_between_rect[i], in_between_rect[j], (0, 0, 255), 2)
        
        # Find Homography
        src_rect  = np.array([[0, height, 1], [0, 0, 1], [width, 0, 1], [width, height, 1]])
        dest_rect = np.array([[x,y,1] for (x,y) in in_between_rect])
        h, status = cv2.findHomography(src_rect, dest_rect)
        hi, status = cv2.findHomography(dest_rect, src_rect)
        return h, hi, marked_img, in_between_rect
    else:
        return None, None, marked_img, None
    
def build_traindata(
    input_img_paths, 
    detector, 
    img_w, img_h, 
    marked_dir, 
    train_dir, 
    resize_size, 
    show_gui=True,
    use_legacy_rect_finding=False,
    ):
    global g_img_resize_factor, top_left_corner, bottom_right_corner
    
    if len(top_left_corner) == 0:
        print("No objects marked!")
        return
    
    warped_inner_rect_cornerss = []
    for i in range(0, len(bottom_right_corner)):
        # Clockwise corner point lists starting at top left for all marked rects
        warped_inner_rect_cornerss.append([top_left_corner[i], (bottom_right_corner[i][0], top_left_corner[i][1]), bottom_right_corner[i], (top_left_corner[i][0], bottom_right_corner[i][1])])
    # Flatten
    warped_inner_rect_corners = [item for sublist in warped_inner_rect_cornerss for item in sublist]
    
    # Iterate through other images
    num_success_imgs = 0
    num_fail_imgs = 0
    for other_img_path in input_img_paths[1:]:
        print(f"Building {other_img_path}...")
        other_img = cv2.imread(other_img_path)
        other_img = resize_img_by_factor(other_img, g_img_resize_factor)
        # Get homography if possible
        h, hi, marked_img, in_between_rect = find_homography_from_aruco(other_img.copy(), detector, img_w, img_h, use_legacy_rect_finding)
        if h is None:
            print("Didn't find the aruco frame :/")
            cv2.imwrite(str(marked_dir / ("marked_error_" + Path(other_img_path).stem + ".png")), marked_img)
            num_fail_imgs+=1
            continue
        num_success_imgs+=1
        
        # Mark found rectangles in inner_rect
        ircs = apply_homography(warped_inner_rect_corners, h)
        draw_img = marked_img.copy()
        for i in range(0, len(ircs)):
            if i % 4 == 3:
                cv2.line(draw_img, ircs[i], ircs[i-3], (0,0,255), 2)
            else:
                cv2.line(draw_img, ircs[i], ircs[i+1], (0,0,255), 2)
        cv2.imwrite(str(marked_dir / ("marked_" + Path(other_img_path).stem + ".png")), draw_img)
        if show_gui:
            cv2.imshow(window_name, draw_img)
            cv2.waitKey(1)
        
        # Compute in_between_rect bounds and crop image
        inner_bounds_x, inner_bounds_y, inner_bounds_w, inner_bounds_h, inner_bounds_xe, inner_bounds_ye = get_bounds(in_between_rect)
        crop_img = other_img[inner_bounds_y:inner_bounds_ye, inner_bounds_x:inner_bounds_xe]
        # Compute bbox coordinate types
        out_img_size = (inner_bounds_w, inner_bounds_h)
        circs = [(x[0] - inner_bounds_x, x[1] - inner_bounds_y) for x in ircs] # cropped-inner-rect-corners
        if resize_size > 0:
            # Resize and pad image
            crop_img, new_size, top, left = resize_and_pad(crop_img, resize_size)
            # Resize and pad bbox points with image
            circs = mult_by_point(circs, (new_size[1] / inner_bounds_w, new_size[0] / inner_bounds_h))
            circs = add_by_point(circs, (left, top))
            out_img_size = (resize_size, resize_size)
        gcircs = unflatten(circs, 4)
        seg_image = np.zeros(tuple(reversed(out_img_size)) + (3,), dtype = np.uint8)
        lircs = [[int(x[0]), int(x[1])] for x in circs]
        glircs = unflatten(lircs, 4) # grouped-listed-normalized-cropped-inner-rect-corners
        for seg_rect in glircs:
            seg_vertecies = np.array(seg_rect)
            cv2.fillPoly(seg_image, pts=[seg_vertecies], color=(255, 255, 255))
        
        # Write out data to files
        cv2.imwrite(str(train_dir / (Path(other_img_path).stem + "_in.png")), crop_img)
        cv2.imwrite(str(train_dir / (Path(other_img_path).stem + "_seg.png")), seg_image)
        with open(train_dir / (Path(other_img_path).stem + "_vertices.txt"), "w") as text_file:
            text_file.write(str(gcircs))
                
    print(f"Successfully vs not successfully annotated imgs: {num_success_imgs}/{num_fail_imgs}")

if __name__ == '__main__':
    main()