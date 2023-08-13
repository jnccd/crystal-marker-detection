import argparse
import glob
from math import isnan
import os
import shutil
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

from utility import *

def main():
    parser = argparse.ArgumentParser(prog='yolov5-gen-evaluation-data', description='Generate testable evaluation data for yolov5 output on some datasets valdata.')
    parser.add_argument('-r','--run', type=str, default='', help='Yolov5 run foldername, or path to runfolder.')
    parser.add_argument('-df','--dataset-folder', type=str, default='',  help='The trainings data folder name to learn from or build into.')
    parser.add_argument('-us','--use-sahi', action='store_true', help='Use Sahi for inference.')
    args = parser.parse_args()
    
    # Get chosen or last yolov5 run dir
    root_dir = Path(__file__).resolve().parent
    train_dir = root_dir / f'../training/yolov5'
    if args.run == '':
        for dir in glob.iglob('training/yolov5/*', recursive=False):
            last_dir = dir
        last_dir = Path(last_dir)
        args.run = last_dir.stem
        print(args.run)
        network_file = last_dir / 'weights/best.pt'
    elif Path(args.run).is_dir():
        train_dir = Path(args.run).parent
        network_file = Path(args.run) / 'weights/best.pt'
        args.run = Path(args.run).stem
    else:
        network_file = train_dir / f'{args.run}/weights/best.pt'
    
    print("Generating evaldata for:", network_file)

    # Torch hub cache support on
    os.system('mkdir ./.cache')
    os.environ['TORCH_HOME'] = './.cache'
    
    print("network_file:",network_file)
    if not args.use_sahi:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=network_file)
    else:
        # Get sahi imports and model if set
        from sahi.predict import get_sliced_prediction
        from sahi import AutoDetectionModel
        model = AutoDetectionModel.from_pretrained(model_type='yolov5', model_path=network_file)
    
    valdata_imgs_path = Path(args.dataset_folder) / 'val/images'
    valdata_labels_path = Path(args.dataset_folder) / 'val/labels'
    
    valdata_imgs_paths = get_files_from_folders_with_ending([valdata_imgs_path], (".png", ".jpg"))
    valdata_labels_paths = get_files_from_folders_with_ending([valdata_labels_path], (".txt"))

    i=0
    out_testdata_path = create_dir_if_not_exists(train_dir / f'{args.run}/test')
    for img_path, label_path in zip(valdata_imgs_paths, valdata_labels_paths):
        
        # Write input picture
        shutil.copyfile(img_path, out_testdata_path / f'{i}_input.png')
        
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        if not args.use_sahi:
            # Default Yolov5 inference
            
            # Write model out
            results = model(img)
            res_df = results.pandas().xyxy[0]
            out_path = out_testdata_path / f'{i}_network_output.txt'
            with open(out_path, "w") as text_file:
                res_df = res_df.reset_index()
                for index, row in res_df.iterrows():
                    if row['confidence'] > 0.5:
                        text_file.write(f"{row['xmin']} {row['ymin']} {row['xmax']} {row['ymax']} {row['confidence']}\n")
            #print(out_img_path)
            cv2.imwrite(str(out_testdata_path / f'{i}_result_render.png'), np.squeeze(results.render()))
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
            result.export_visuals(export_dir=str(out_testdata_path), file_name=f'{i}_result_render')
            out_path = out_testdata_path / f'{i}_network_output.txt'
            with open(out_path, "w") as text_file:
                for pred in result.object_prediction_list:
                    if pred.score.value > 0.5:
                        text_file.write(f"{pred.bbox.minx} {pred.bbox.miny} {pred.bbox.maxx} {pred.bbox.maxy} {pred.score.value}\n")
            
        # Write labels
        # Rasterize Segmentation image
        sanity_check_image = np.zeros((img_w, img_h) + (3,), dtype = np.uint8)
        with open(label_path, 'r') as file:
            vd_bbox_lines = file.read().split('\n')
        vd_bbox_lines_og = vd_bbox_lines
        #print("vd_bbox_lines", vd_bbox_lines)
        vd_bbox_lines = list(filter(lambda s: s and not s.isspace(), vd_bbox_lines)) # Filter whitespace lines away
        target_output_path = out_testdata_path / f'{i}_target_output.txt'
        #print("target_output_path", target_output_path)
        #print("vd_bbox_lines", vd_bbox_lines)
        #print("label_path", label_path)
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
                #print(f"{min_x} {min_y} {max_x} {max_y}")
                
                verts = np.array([(int(min_x), int(min_y)), (int(min_x), int(max_y)), (int(max_x), int(max_y)), (int(max_x), int(min_y))])
                #print(verts)
                cv2.fillPoly(sanity_check_image, pts=[verts], color=(255, 255, 255))
                cv2.imwrite(str(out_testdata_path / f'{i}_target_output.png'), sanity_check_image)
                
        i+=1

if __name__ == '__main__':
    main()