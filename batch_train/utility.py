import os
from pathlib import Path
import shutil
from math import isnan
import cv2
import numpy as np

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

def write_textfile(text, tf_path):
    with open(tf_path, "w") as text_file:
        text_file.write(text)
        
def get_files_from_folders_with_ending(folders, ending):
    paths = []
    for folder in folders:
        paths.extend(sorted(
            [
                os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(ending)
            ]
        ))
    return paths

def create_dir_if_not_exists(dir: Path, clear = False):
    if clear and os.path.isdir(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def handle_model_out(
    i,
    boxes, 
    img_w, 
    img_h, 
    out_testdata_path,
    label_path,
    confidence_threshold, 
    border_ignore_size, 
    squareness_threshold,
    build_debug_output = False,
    mask = None,
    ):
    
    # Filter model out
    #print('boxes', boxes)
    if border_ignore_size > 0:
        boxes = list(filter(lambda box: #xmin, ymin, xmax, ymax, conf: 
            box[0] / img_w > border_ignore_size and 
            box[1] / img_h > border_ignore_size and
            1 - (box[2] / img_w) > border_ignore_size and 
            1 - (box[3] / img_h) > border_ignore_size, boxes))
    if squareness_threshold > 0:
        boxes = list(filter(lambda box: ((box[2] - box[0]) / (box[3] - box[1]) if (box[2] - box[0]) / (box[3] - box[1]) < 1 else 1 / ((box[2] - box[0]) / (box[3] - box[1]))) > squareness_threshold, boxes))
    boxes = list(filter(lambda box: box[4] > confidence_threshold, boxes))
    if mask is not None:
        box_windows = [mask[box[1]:box[3], box[0]:box[2]] for box in boxes]
        print(mask.shape, box_windows, [np.max(x) for x in box_windows])
        boxes = list(filter(lambda box: np.average(mask[box[1]:box[3], box[0]:box[2]]) > 80, boxes))
    
    # Rasterize Segmentation image
    if build_debug_output:
        sanity_check_image = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
        for min_x, min_y, max_x, max_y, conf in boxes:
            verts = np.array([(int(min_x), int(min_y)), (int(min_x), int(max_y)), (int(max_x), int(max_y)), (int(max_x), int(min_y)), (int(min_x), int(min_y))])
            cv2.fillPoly(sanity_check_image, pts=[verts], color=(255, 255, 255))
        cv2.imwrite(str(out_testdata_path / f'{i}_network_output.png'), sanity_check_image)
    # Write model out
    with open(out_testdata_path / f'{i}_network_output.txt', "w") as text_file:
        for xmin, ymin, xmax, ymax, conf in boxes:
            text_file.write(f"{xmin} {ymin} {xmax} {ymax} {conf}\n")
        
    # Write labels
    # Rasterize Segmentation image
    sanity_check_image = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
    with open(label_path, 'r') as file:
        vd_bbox_lines = file.read().split('\n')
    vd_bbox_lines_og = vd_bbox_lines
    vd_bbox_lines = list(filter(lambda s: s and not s.isspace(), vd_bbox_lines)) # Filter whitespace lines away
    target_output_path = out_testdata_path / f'{i}_target_output.txt'
    with open(target_output_path, "w") as text_file:
        for line in vd_bbox_lines:
            sc, sx, sy, sw, sh = line.split(' ')
            
            if any(isnan(float(x)) for x in [sx, sy, sw, sh]):
                print(f'Encountered NaN output in {label_path}', list(vd_bbox_lines), vd_bbox_lines_og, sx, sy, sw, sh)
                continue
            
            bbox_w = float(sw) * img_w
            bbox_h = float(sh) * img_h
            min_x = float(sx) * img_w - bbox_w / 2
            min_y = float(sy) * img_h - bbox_h / 2
            max_x = bbox_w + min_x
            max_y = bbox_h + min_y
            
            text_file.write(f"{min_x} {min_y} {max_x} {max_y}\n")
            
            if build_debug_output:
                verts = np.array([(int(min_x), int(min_y)), (int(min_x), int(max_y)), (int(max_x), int(max_y)), (int(max_x), int(min_y))])
                cv2.fillPoly(sanity_check_image, pts=[verts], color=(255, 255, 255))
                cv2.imwrite(str(out_testdata_path / f'{i}_target_output.png'), sanity_check_image)