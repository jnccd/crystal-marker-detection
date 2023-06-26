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

window_name = "Window"

top_left_corner=[]
bottom_right_corner=[]
new_top_left = None
cur_m_pos = None

max_img_width = 1920
max_img_height = 1080
g_img_resize_factor = 1

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
    
def apply_homography(point2D_list, h, convert_to_int = True):
    hps = [h @ (p[0], p[1], 1) for p in point2D_list] 
    ps = [(p[0] / p[2], p[1] / p[2]) for p in hps]
    
    if convert_to_int:
        return [(int(p[0]), int(p[1])) for p in ps]
    else:
        return ps
    
def get_bounds(point2D_list):
    x = min([p[0] for p in point2D_list])
    y = min([p[1] for p in point2D_list])
    xe = max([p[0] for p in point2D_list])
    ye = max([p[1] for p in point2D_list])
    w = xe - x
    h = ye - y
    return x, y, w, h, xe, ye

def add_by_point(point2D_list, a):
    return [(p[0] + a[0], p[1] + a[1]) for p in point2D_list]

def mult_by_point(point2D_list, m):
    return [(p[0] * m[0], p[1] * m[1]) for p in point2D_list]

def divide_by_point(point2D_list, d):
    return [(p[0] / d[0], p[1] / d[1]) for p in point2D_list]

def unflatten(list, chunk_size):
    return [list[n:n+chunk_size] for n in range(0, len(list), chunk_size)]

def resize_and_pad(img: Mat, desired_size: int):
    old_size = img.shape[:2]

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    rimg = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    brimg = cv2.copyMakeBorder(rimg, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    return brimg, new_size, top, left

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

def resize_img_by_factor(img, factor):
    img_h, img_w = img.shape[:2]
    target_size = (int(img_w * factor), int(img_h * factor))
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def keep_image_size_in_check(img):
    global max_img_width, max_img_height
    
    img_h, img_w = img.shape[:2]
    if img_w > max_img_width:
        img = set_img_width(img, max_img_width)
    if img_h > max_img_height:
        img = set_img_height(img, max_img_height)
        
    return img
    
def segment_img_between_poly_labels(img, polys, dim: Literal[0,1], collage_padding = 5):
    img_h, img_w = img.shape[:2]
    
    if dim == 0:
        polys = sorted(polys, key=lambda x: x.centroid.x)
    else:
        polys = sorted(polys, key=lambda x: x.centroid.y)
        
    segments = []
    for i in range(len(polys)-1):
        upper_poly_lower_bound = polys[i].bounds[2 if dim == 0 else 3]
        lower_poly_upper_bound = polys[i+1].bounds[0 if dim == 0 else 1]
        dim_distance = lower_poly_upper_bound - upper_poly_lower_bound
        if dim_distance > collage_padding:
            if len(segments) > 0:
                last_end = segments[-1]['end']
            else:
                last_end = 0
            end = int((upper_poly_lower_bound + lower_poly_upper_bound) / 2)
            if dim == 0:
                corners = list(filter(lambda x: x.exterior.coords[0][0] > last_end and x.exterior.coords[0][0] < end, polys))
                seg_img = img[0:img_w, last_end:end]
            else:
                corners = list(filter(lambda x: x.exterior.coords[0][1] > last_end and x.exterior.coords[0][1] < end, polys))
                seg_img = img[last_end:end, 0:img_h]
            segments.append({   'end': end,
                                'beginning': last_end,
                                'size': end - last_end,
                                'poly_index': i,
                                'corners': corners,
                                'img': seg_img,
                                })
    if len(segments) > 0:
        last_end = segments[-1]['end']
    else: 
        last_end = 0
    end = img_w
    if dim == 0:
        corners = list(filter(lambda x: x.exterior.coords[0][0] > last_end and x.exterior.coords[0][0] < end, polys))
        seg_img = img[0:img_w, last_end:end]
    else:
        corners = list(filter(lambda x: x.exterior.coords[0][1] > last_end and x.exterior.coords[0][1] < end, polys))
        seg_img = img[last_end:end, 0:img_h]
    segments.append({ 
        'end': end,
        'beginning': last_end,
        'size': end - last_end,
        'poly_index': len(polys)-1,
        'corners': corners,
        'img': seg_img,
        })
    
    return segments

def rebuild_img_from_segments(segments, out_img_size_wh, dim: Literal[0,1]):
    aug_image = np.zeros(tuple(reversed(out_img_size_wh)) + (3,), dtype = np.uint8)
    aug_polys = []
    pos = 0
    for seg in segments:
        if dim == 0: # x
            aug_image[0:out_img_size_wh[1],pos:pos+seg['size']] = seg['img']
        else: # y
            aug_image[pos:pos+seg['size'],0:out_img_size_wh[0]] = seg['img']
            
        #print('y_pos',y_pos)
        #print('seg[beginning]',seg['beginning'])
        
        print(dim, 'seg corners',seg['corners'])
        for poly_corners in seg['corners']:
            if dim == 0:
                aug_polys.append(Polygon([(x[0] - seg['beginning'] + pos, x[1]) for x in poly_corners.exterior.coords]))
            else:
                aug_polys.append(Polygon([(x[0], x[1] - seg['beginning'] + pos) for x in poly_corners.exterior.coords]))
        pos += seg['size']
    
    #cv2.imwrite(str(train_dir / (Path(other_img_path).stem + "_aug.png")), aug_image)
    #print('aug_gcircs',len(aug_gcircs),aug_gcircs)
    #print('gcircs',len(gcircs),gcircs)
    
    return aug_image, aug_polys
    
def build_traindata(
    input_img_paths, 
    detector, 
    img_w, img_h, 
    marked_dir, 
    train_dir, 
    resize_size, 
    show_gui=True,
    collage_augment=False,
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
        # TODO: Add augmentation code here? Polygon intersection? Move before resize?
        # --- Augment -------------------------------------------------------------------------
        if collage_augment:
            polys = []
            for gcirc in gcircs:
                polys.append(Polygon(gcirc))
            
            # segment in y first
            segments_y = segment_img_between_poly_labels(crop_img, polys, 1)
            for seg_y in segments_y:
                segs_x = segment_img_between_poly_labels(seg_y['img'], seg_y['corners'], 0)
                random.shuffle(segs_x)
                
                seg_img_h, seg_img_w = seg_y['img'].shape[:2]
                segs_img, segs_polys = rebuild_img_from_segments(segs_x, (seg_img_w, seg_img_h), 0)
                
                seg_y['img'] = segs_img
                seg_y['corners'] = segs_polys
            random.shuffle(segments_y)
            aug_img, aug_polys = rebuild_img_from_segments(segments_y, out_img_size, 1)
            
            crop_img = aug_img
            gcircs = [[(point[0], point[1]) for point in poly.exterior.coords[:-1]] for poly in aug_polys] # Convert polys back to corner points list
            circs = [item for sublist in gcircs for item in sublist] # write aug gcircs back into circs
            
            print(gcircs)
            
        # -------------------------------------------------------------------------------------
        bgcircs = [get_bounds(x) for x in gcircs] # boundsOf-grouped-cropped-inner-rect-corners
        
        # Normalize coordinates
        ncircs = divide_by_point(circs, out_img_size)
        gncircs = unflatten(ncircs, 4)
        bgncircs = [get_bounds(x) for x in gncircs] #boundsOf-grouped-normalized-cropped-inner-rect-corners
        # Rasterize Segmentation image
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
            for rect in gcircs:
                text_file.write(f"{rect[0]}, {rect[1]}, {rect[2]}, {rect[3]}\n")
        # xywh formats
        with open(train_dir / (Path(other_img_path).stem + "_xywh.txt"), "w") as text_file:
            for bounds in bgcircs:
                text_file.write(f"{bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}\n")
        with open(train_dir / (Path(other_img_path).stem + "_xywh_n.txt"), "w") as text_file:
            for bounds in bgncircs:
                text_file.write(f"{bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}\n")
        # xyxy formats
        with open(train_dir / (Path(other_img_path).stem + "_xyxy.txt"), "w") as text_file:
            for bounds in bgcircs:
                text_file.write(f"{bounds[0]} {bounds[1]} {bounds[4]} {bounds[5]}\n")
        with open(train_dir / (Path(other_img_path).stem + "_xyxy_n.txt"), "w") as text_file:
            for bounds in bgncircs:
                text_file.write(f"{bounds[0]} {bounds[1]} {bounds[4]} {bounds[5]}\n")
        # center xy for coco
        with open(train_dir / (Path(other_img_path).stem + "_cxcywh_n.txt"), "w") as text_file:
            for bounds in bgncircs:
                text_file.write(f"{bounds[0]+(bounds[2]/2)} {bounds[1]+(bounds[3]/2)} {bounds[2]} {bounds[3]}\n")
                
    print(f"Successfully vs not successfully annotated imgs: {num_success_imgs}/{num_fail_imgs}")

def main():
    global window_name, top_left_corner, bottom_right_corner, new_top_left, cur_m_pos, g_img_resize_factor
    
    parser = argparse.ArgumentParser(prog='traindata-creator', description='Creates traindata in bulk for image series on marked planes.')
    parser.add_argument('-if','--input-folder', type=str, help='The path to the folder containing an image series.')
    parser.add_argument('-s','--size', type=int, default=0, help='The width and height of the traindata images.')
    parser.add_argument('-ng','--no-gui', action='store_true', help='Builds traindata immediately based on cached label data.')
    parser.add_argument('-a','--augment', action='store_true', help='Augment the training data is some way.')
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
    dataseries_dir = root_dir / 'dataseries' / f'{str(args.size)}-{"aug-" if args.augment else ""}{input_dir.name}'
    marked_dir = dataseries_dir / 'images_marked'
    if not os.path.exists(marked_dir):
        os.makedirs(marked_dir)
    train_dir = dataseries_dir / 'images_traindata'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    # Load first image and preprocess
    img = cv2.imread(input_img_paths[0])
    l_img_h, l_img_w = img.shape[:2]
    # Resize image to displayable sizes if necessary (neural network inputs are so small compared to this that it should not matter and is more convenient for displaying and for performance)
    img = keep_image_size_in_check(img) 
    img_h, img_w = img.shape[:2]
    g_img_resize_factor = img_w / l_img_w
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
            collage_augment=args.augment,
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
                collage_augment=args.augment,
                )

        # Draw
        display_img = warped_img.copy()
        for i in range(0, len(bottom_right_corner)):
            cv2.rectangle(display_img, top_left_corner[i], bottom_right_corner[i], (0,255,0), 1, 8)
        if new_top_left is not None:
            cv2.rectangle(display_img, new_top_left, cur_m_pos, (0,255,0), 1, 8)
        cv2.imshow(window_name, display_img)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()