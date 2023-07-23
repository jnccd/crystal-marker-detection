import argparse
import ast
import json
from math import isnan
import os
import shutil
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from utils import *

def main():
    # Parse
    parser = argparse.ArgumentParser(prog='yolov8-gen-evaluation-data', description='Generate testable evaluation data for yolov8 output on some datasets valdata.')
    parser.add_argument('-r','--run-folder', type=str, default='', help='Yolov5 run foldername, or path to runfolder.')
    parser.add_argument('-df','--dataset-folder', type=str, default='',  help='The trainings data folder name to learn from or build into.')
    args = parser.parse_args()
    
    # Get chosen or last yolov5 run dir
    train_dir = Path(args.run_folder)
    network_file = Path(args.run_folder) / 'weights/best.pt'
    model = YOLO(network_file)
    
    gen_evaldata(model, args.dataset_folder, create_dir_if_not_exists(train_dir / 'test'))
    

def gen_evaldata(model, valset_path, out_testdata_path):
    valdata_imgs_path = Path(valset_path) / 'val/images'
    valdata_labels_path = Path(valset_path) / 'val/labels'
    
    valdata_imgs_paths = get_files_from_folders_with_ending([valdata_imgs_path], (".png", ".jpg"))
    valdata_labels_paths = get_files_from_folders_with_ending([valdata_labels_path], (".txt"))
    
    results = model(valdata_imgs_path / '*.png')
    
    for i, (img_path, label_path) in enumerate(zip(valdata_imgs_paths, valdata_labels_paths)):
        
        # Write input picture
        shutil.copyfile(img_path, out_testdata_path / f'{i}_input.png')
        
        # Write model out
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        result = results[i]
        out_path = out_testdata_path / f'{i}_network_output.txt'
        with open(out_path, "w") as text_file:
            for box in result.boxes:
                if box.conf > 0.5:
                    text_line_numbers = [float(x) for x in list(box.xyxy[0]) + [box.conf]]
                    #print(i, text_line_numbers)
                    text_file.write(f"{' '.join([str(x) for x in text_line_numbers])}\n")
        cv2.imwrite(str(out_testdata_path / f'{i}_result_plot.png'), np.squeeze(result.plot()))
        
        # Write labels and Rasterize label Segmentation image
        sanity_check_image = np.zeros((img_w, img_h) + (3,), dtype = np.uint8)
        with open(label_path, 'r') as file:
            vd_bbox_lines = file.read().split('\n')
        vd_bbox_lines = filter(lambda s: s and not s.isspace(), vd_bbox_lines) # Filter whitespace lines away
        
        target_output_path = out_testdata_path / f'{i}_target_output.txt'
        with open(target_output_path, "w") as text_file:
            for line in vd_bbox_lines:
                sc, sx, sy, sw, sh = line.split(' ')
                
                if any(isnan(float(x)) for x in [sx, sy, sw, sh]):
                    print(f'Encountered NaN output in {target_output_path}')
                    continue
                
                bbox_w = float(sw) * img_w
                bbox_h = float(sh) * img_h
                min_x = float(sx) * img_w - bbox_w / 2
                min_y = float(sy) * img_h - bbox_h / 2
                max_x = bbox_w + min_x
                max_y = bbox_h + min_y
                
                text_file.write(f"{min_x} {min_y} {max_x} {max_y}\n")
                
                verts = np.array([(int(min_x), int(min_y)), (int(min_x), int(max_y)), (int(max_x), int(max_y)), (int(max_x), int(min_y))])
                #print(verts)
                cv2.fillPoly(sanity_check_image, pts=[verts], color=(255, 255, 255))
        cv2.imwrite(str(out_testdata_path / f'{i}_target_output.png'), sanity_check_image)
        
if __name__ == '__main__':
    main()