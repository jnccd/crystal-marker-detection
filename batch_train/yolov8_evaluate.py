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

from utility import *

def main():
    # Parse
    parser = argparse.ArgumentParser(prog='yolov8-evaluate', description='Generate testable evaluation data for yolov8 output on some datasets valdata.')
    parser.add_argument('-r','--run-folder', type=str, help='Path to a Yolov8 run folder to evaluate the output from.')
    parser.add_argument('-t','--testset-folder', type=str, help='The dataset to use the valdata of as a testset for this evaluation.')
    parser.add_argument('-tn','--test-folder-name', type=str, default='test', help='The folder name for this test in the run-folder, per default its "test".')
    
    parser.add_argument('-ct','--confidence-threshold', type=float, default=0.5, help='The minimum confidence of considered predictions.')
    parser.add_argument('-bis','--border-ignore-size', type=float, default=0, help='Ignore markers at the border of the image, given in widths from 0 to 0.5.')
    parser.add_argument('-sqt','--squareness-threshold', type=float, default=0, help='The minimum squareness of considered prediction boxes.')
    parser.add_argument('-us','--use-sahi', action='store_true', help='Use Sahi for inference.')
    parser.add_argument('-dbo','--debug-output-imgs', action='store_true', help='Generate more output.')
    args = parser.parse_args()
    
    # Set up Paths
    run_folder_path = Path(args.run_folder)
    run_test_folder_path = create_dir_if_not_exists(run_folder_path / args.test_folder_name, clear=True)
    run_best_model_path = run_folder_path / 'weights/best.pt'
    
    # Generate evaldata
    print("Generating evaldata for:", run_best_model_path)
    gen_evaldata(
        model_path= run_best_model_path, 
        valset_path= args.testset_folder, 
        out_testdata_path= run_test_folder_path,
        
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
        model = YOLO(model_path)
        results = model(valdata_imgs_path / '*.png')
    else:
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
            for box in result.boxes:
                #print('box.xyxy', box.xyxy)
                boxes.append(tuple([x.item() for x in box.xyxy[0][:4]] + [box.conf.item()]))
            cv2.imwrite(str(out_testdata_path / f'{i}_result_plot.png'), np.squeeze(result.plot()))
        else:
            # Sahi inference
            result = get_sliced_prediction(
                img_path,
                model,
                slice_height = 640,
                slice_width = 640,
                overlap_height_ratio = 0.2,
                overlap_width_ratio = 0.2
            )
            for pred in result.object_prediction_list:
                # print(f'pred {[x.cpu().item() for x in pred.bbox.to_xyxy()]}')
                boxes.append((pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy, pred.score.value))
            result.export_visuals(export_dir=str(out_testdata_path), file_name=f'{i}_result_render')
            
        print(f'boxes {boxes}')
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
        
if __name__ == '__main__':
    main()