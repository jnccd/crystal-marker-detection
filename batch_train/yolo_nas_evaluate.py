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

import torch

if not("SUPER_GRADIENTS_LOG_DIR" in os.environ.keys()):
    os.environ["SUPER_GRADIENTS_LOG_DIR"] = "./training/sg_logs"
from super_gradients.training import models
from super_gradients.training.utils.checkpoint_utils import adaptive_load_state_dict
from super_gradients.training.utils.predict import ImagesDetectionPrediction

from utils import *

def main():
    # Parse
    parser = argparse.ArgumentParser(prog='yolov-nas-evaluate', description='Evaluate yolo nas output on a datasets valdata.')
    parser.add_argument('-r','--run-folder', type=str, help='Path to a Yolo Nas run folder.')
    parser.add_argument('-t','--testset-folder', type=str, help='The dataset to use the valdata of as a testset for this evaluation.')
    parser.add_argument('-tn','--test-folder-name', type=str, default='test', help='The folder name for this test in the run-folder, per default its "test".')
    parser.add_argument('-mt','--model-type', type=str, default='yolo_nas_s', help='The type of model that should be loaded.')
    
    parser.add_argument('-ct','--confidence-threshold', type=float, default=0.5, help='The minimum confidence of considered predictions.')
    parser.add_argument('-bis','--border-ignore-size', type=float, default=0, help='Ignore markers at the border of the image, given in widths from 0 to 0.5.')
    parser.add_argument('-sqt','--squareness-threshold', type=float, default=0, help='The minimum squareness of considered prediction boxes.')
    parser.add_argument('-us','--use-sahi', action='store_true', help='Use Sahi for inference.')
    parser.add_argument('-dbo','--debug-output-imgs', action='store_true', help='.')
    args = parser.parse_args()
    
    # Set up Paths
    run_folder_path = Path(args.run_folder)
    run_test_folder_path = create_dir_if_not_exists(run_folder_path / args.test_folder_name, clear=True)
    run_best_model_path = str(list(run_folder_path.glob('**/*_best.pth'))[0])
    
    # Generate evaldata
    print("Generating evaldata for:", run_best_model_path)
    gen_evaldata(
        model_path= run_best_model_path, 
        valset_path= args.testset_folder, 
        out_testdata_path= run_test_folder_path,
        model_type= args.model_type,
        
        confidence_threshold= args.confidence_threshold,
        squareness_threshold= args.squareness_threshold,
        use_sahi= args.use_sahi,
        border_ignore_size= args.border_ignore_size,
        build_debug_output=args.debug_output_imgs,
    )
    
    # Start analyze script
    os.system(f'python evaluation/analyze.py -av {run_test_folder_path}')
    
def gen_evaldata(
    model_path,
    valset_path, 
    out_testdata_path, 
    model_type = 'yolo_nas_s',
    
    confidence_threshold = 0.5,
    squareness_threshold = 0,
    use_sahi = False,
    border_ignore_size = 0,
    build_debug_output: bool = False
    ):
    valdata_imgs_path = Path(valset_path) / 'val/images'
    valdata_labels_path = Path(valset_path) / 'val/labels'
    
    valdata_imgs_paths = get_files_from_folders_with_ending([valdata_imgs_path], (".png", ".jpg"))
    valdata_labels_paths = get_files_from_folders_with_ending([valdata_labels_path], (".txt"))
    
    if not use_sahi:
        print(f'Loading model from {model_path}...')
        model = models.get(
            model_type,
            num_classes=1, 
            checkpoint_path=model_path
        )
        
        model.eval()
        test_imgs = get_files_from_folders_with_ending([Path(valset_path) / 'val' / 'images'], '.png')
        results: ImagesDetectionPrediction = model.predict(test_imgs)
    else:
        raise NotImplementedError()
        # Get sahi imports and model if set
        from sahi.predict import get_sliced_prediction
        from sahi import AutoDetectionModel
        model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=model_path)
    
    for i, (img_path, label_path) in enumerate(zip(valdata_imgs_paths, valdata_labels_paths)):
        
        # Write input picture
        shutil.copyfile(img_path, out_testdata_path / f'{i}_input.png')
        
        # Get inference img
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # Gather inference results from different sources
        boxes = [] # [(xmin, ymin, xmax, ymax, conf), ..]
        if not use_sahi:
            result = results[i]
            
            for i, (label, conf, bbox) in enumerate(zip(result.prediction.labels, 
                                                        result.prediction.confidence, 
                                                        result.prediction.bboxes_xyxy)):
                boxes.append(tuple(list(bbox)[:4] + [conf]))
            # TODO: result.save
        else:
            raise NotImplementedError()
        print('boxes', boxes)
            
        handle_model_out(
            i, 
            boxes,
            img_w, 
            img_h, 
            out_testdata_path,
            label_path,
            confidence_threshold, 
            border_ignore_size, 
            squareness_threshold,
            build_debug_output
            )
        
    # Add test def dict
    valset_def_dict = json.loads(read_textfile(Path(valset_path) / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    write_textfile(json.dumps({
            'model_path': str(model_path),
            'valset': valset_def_dict,
            'confidence_threshold': confidence_threshold,
            'use_sahi': use_sahi,
            'border_ignore_size': border_ignore_size,
            'squareness_threshold': squareness_threshold,
            'build_debug_output': build_debug_output,
        }, indent=4), out_testdata_path / '_test-def.json')
        
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
    build_debug_output
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
        
if __name__ == '__main__':
    main()