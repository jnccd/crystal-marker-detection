import math
from pathlib import Path
import random
import cv2
import numpy as np
from shapely import box, LineString, normalize, Polygon, Point, intersection, intersection_all

from utils import *

def eelongate(l: LineString, mult: float):
    x, y = l.xy
    x_diff = x[1] - x[0]
    y_diff = y[1] - y[0]
    return LineString([(x[0] - x_diff * mult, y[0] - y_diff * mult), (x[1] + x_diff * mult, y[1] + y_diff * mult)])

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    root_dir = Path(__file__).resolve().parent
    output_folder = create_dir_if_not_exists(root_dir / 'output/pt-seg')
    eval_folder = create_dir_if_not_exists(output_folder / 'eval')
    to_rect_output_folder = create_dir_if_not_exists(root_dir / 'output/to-rect')
    marker_img_path = root_dir / 'assets/in-img-marker.png'
    
    marker_img = cv2.imread(str(marker_img_path),0)
    
    pred_img_paths = [Path(x) for x in get_files_from_folders_with_ending([eval_folder], '_pred.png')]
    in_img_paths = [Path(x) for x in get_files_from_folders_with_ending([eval_folder], '_in.png')]
    for pred_img_path, in_img_path in zip(pred_img_paths, in_img_paths):
        in_img = cv2.imread(str(in_img_path))
        in_img_h, in_img_w = in_img.shape[:2]
        pred_img = cv2.imread(str(pred_img_path))
        pred_img_h, pred_img_w = pred_img.shape[:2]
        
        gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        
        # Canny Edge
        (T, img_thresh) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        img_edges = cv2.Canny(img_thresh, 50, 300, None, 5)
        # Prevent canny from predicting edges near image edge
        edge_border_size_x = int(in_img_w/8)
        edge_border_size_y = int(in_img_h/8)
        img_edges = cv2.rectangle(img_edges, (0, 0), (edge_border_size_x, in_img_h), (0,0,0), thickness=-1)
        img_edges = cv2.rectangle(img_edges, (0, 0), (in_img_w, edge_border_size_y), (0,0,0), thickness=-1)
        img_edges = cv2.rectangle(img_edges, (in_img_w - edge_border_size_x, 0), (in_img_w, in_img_h), (0,0,0), thickness=-1)
        img_edges = cv2.rectangle(img_edges, (0, in_img_h - edge_border_size_y), (in_img_w, in_img_h), (0,0,0), thickness=-1)
        # ---
        cv2.imwrite(str(to_rect_output_folder / f'{pred_img_path.stem}_canny_edges.png'), img_edges)
        cv2.imshow('image',img_edges)
        cv2.waitKey(0)
        # Hough Transform
        linesP = cv2.HoughLinesP(img_edges, 1, np.pi / 40, 20, None, 1, 10)
        img_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if linesP is not None:
            print(f"Found {len(linesP)} lines")
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(img_draw, (l[0], l[1]), (l[2], l[3]), (0,255,255), 1, cv2.LINE_AA)
        else:
            continue
        # Visualize Lines
        cv2.imwrite(str(to_rect_output_folder / f'{pred_img_path.stem}_hough_lines.png'), img_draw)
        cv2.imshow('image',img_draw)
        cv2.waitKey(0)
        
        if len(linesP) < 4:
            print('not enough lines :c')
            continue
        
        # Interpret Hough Lines
        print(f"lines: {linesP}")
        shapely_lines = [LineString([(l[0][0], l[0][1]), (l[0][2], l[0][3])]) for l in linesP]
        # # Take 4 longest
        # shapely_lines = sorted(shapely_lines, key=lambda x: -x.length)
        # shapely_lines = shapely_lines[:4]
        # Make longer
        shapely_lines = [eelongate(x, 5) for x in shapely_lines]
        print(shapely_lines)
        # Get intersects
        intersect_points = []
        for i in range(len(shapely_lines)):
            for j in range(i + 1, len(shapely_lines)):
                intersect = intersection(shapely_lines[i], shapely_lines[j])
                print(f"intersect {intersect}, {intersect.geom_type}")
                if str(intersect.geom_type) == 'Point': # Yes, this is the best way to check the type
                    #print('is pooint')
                    intersect: Point = intersect # Linting C:
                    intersect_points.append((intersect.centroid.x, intersect.centroid.y))
        print(f'intersect_points {intersect_points}')
        # Remove intersect duplicates 
        i = 0
        while i < len(intersect_points):
            j = i + 1
            while j < len(intersect_points):
                if distance(intersect_points[i], intersect_points[j]) < 5:
                    intersect_points.pop(j)
                    j -= 1
                    if j < i:
                        i -= 1
                j += 1
            i += 1
        print(f'intersect_points {intersect_points}')
        # Visualize Intersects
        for corner in intersect_points:
            cv2.circle(img_draw, (int(corner[0]), int(corner[1])), 5, (255,0,0))
        cv2.imshow('image',img_draw)
        cv2.waitKey(0)
        # Take 4 intersects closest to middle
        img_middle_point = (in_img_w/2, in_img_h/2)
        corners = sorted(intersect_points, key=lambda x: distance(x, img_middle_point))[:4]
        
        print(corners)
        print(f'---{pred_img_path.stem}-----------')
        for c in corners:
            print(c)
        
        # Thresh in img
        in_image_gray = cv2.imread(str(in_img_path),0)
        block_size = 90#int(img_w/600*50)
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        in_image_t = cv2.adaptiveThreshold(in_image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,block_size,3)
        
        # Homography
        src_rect  = np.array([[0, in_img_h, 1], [0, 0, 1], [in_img_w, 0, 1], [in_img_w, in_img_h, 1]])
        dest_rect = np.array([(int(x[0]), int(x[1])) for x in corners])
        hi, status = cv2.findHomography(dest_rect, src_rect)
        marker_area_img = cv2.warpPerspective(in_image_t, hi, (in_img_w, in_img_h))
        # cv2.imshow('image', marker_area_img)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        # Compare markers
        res_marker_img = cv2.resize(marker_img, marker_area_img.shape[:2], interpolation=cv2.INTER_NEAREST_EXACT)
        inv_res_marker_img = cv2.bitwise_not(res_marker_img)
        min_diff = 9999999
        min_i = 0
        concat_imgs = []
        for i, marker_rot in enumerate([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]):
            rot_marker_area_img = cv2.rotate(marker_area_img, marker_rot) if marker_rot is not None else marker_area_img
            
            marker_img_diff = np.abs(rot_marker_area_img.astype('int32') - res_marker_img.astype('int32')).astype('uint8')
            marker_img_diff_sum = np.sum(marker_img_diff)
            inv_marker_img_diff = np.abs(rot_marker_area_img.astype('int32') - inv_res_marker_img.astype('int32')).astype('uint8')
            inv_marker_img_diff_sum = np.sum(inv_marker_img_diff)
            
            mdiff = min(marker_img_diff_sum, inv_marker_img_diff_sum)
            if mdiff < min_diff:
                min_diff = mdiff
                min_i = i
            
            print(mdiff, i, min_i, marker_img_diff_sum, inv_marker_img_diff_sum)
            concat_img = cv2.hconcat([res_marker_img, rot_marker_area_img, marker_img_diff])
            concat_imgs.append(concat_img)
            # cv2.imshow('image', concat_img)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break
        cv2.imwrite(str(to_rect_output_folder / f'{pred_img_path.stem}_roll_compare.png'), cv2.vconcat(concat_imgs))
            
        # Fix point order
        corners = np.roll(corners, min_i, axis=0)
        print('---roll-----------')
        for c in corners:
            print(c)
        
        # Write 
        in_image_grgb = cv2.imread(str(in_img_path))#cv2.cvtColor(in_image_t, cv2.COLOR_GRAY2RGB) #
        pts = np.array([(int(point[0]), int(point[1])) for point in corners], dtype=np.int32)
        in_image_grgb = cv2.polylines(in_image_grgb, pts=[pts], isClosed=True, color=(0,255,0))
        for i, pt in enumerate(pts):
            in_image_grgb = cv2.putText(in_image_grgb, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, 
                  (0,0,255), 2, cv2.LINE_AA, False)
        write_textfile(str([(x[0], x[1]) for x in corners]), to_rect_output_folder / f'{pred_img_path.stem}_p.txt')
        cv2.imwrite(str(to_rect_output_folder / f'{pred_img_path.stem}_rect.png'), in_image_grgb)
        cv2.imshow('image', in_image_grgb)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    main()