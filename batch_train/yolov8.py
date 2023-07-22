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
from ultralytics import YOLO

from utils import *

def main():
    # Parse
    parser = argparse.ArgumentParser(prog='', description='.')
    parser.add_argument('-d','--datasets-path', type=str, default='', help='.')
    parser.add_argument('-v','--testset-path', type=str, default='', help='.')
    
    parser.add_argument('-s','--img-size', type=int, default=640, help='Sets the img size of the model.')
    parser.add_argument('-b','--batch-size', type=int, default=-1, help='Sets the batch size to train with, -1 is yolov8 AutoBatch.')
    parser.add_argument('-e','--epochs', type=int, default=100, help='Sets the epochs to train for.')
    parser.add_argument('-m','--model', type=str, default='yolov8s', help='Sets the model to train with.')
    parser.add_argument('-rw','--init-random-weights', action='store_true', help='.')
    
    args = parser.parse_args()

    # Paths
    root_dir = Path(__file__).resolve().parent
    datasets_path = Path(args.datasets_path)
    datasets_dirs = [x for x in datasets_path.glob('**/yolov5-*/')
                    if x.is_dir() 
                        and not str(x).__contains__("_old") 
                        and not str(x).__contains__("-valset")]
    testset_path = root_dir.parent / args.testset_path
    newline_char = "\n" # Python 3.9 :/
    print(f'Running ensample run using the following datasets:\n{newline_char.join([str(x) for x in datasets_dirs])}')
    
    # Train
    start_time = time.time()

    for dataset_dir in datasets_dirs[:1]:
        yolov8_train_loop(dataset_dir, 
                          testset_path, 
                          run_name=dataset_dir.stem,
                          epochs=args.epochs,
                          img_size=args.img_size,
                          batch_size=args.batch_size,
                          model=args.model,
                          pretrained=not args.init_random_weights)
        
    end_time = time.time()
    diff_time = end_time  - start_time
    print(f'Training took: {time.strftime("%H:%M:%S", time.gmtime(diff_time))}')

def yolov8_train_loop(dataset_path, 
                      valset_path, 
                      run_name = 'default', 
                      img_size = 640, 
                      batch_size = -1, 
                      epochs = 100, 
                      model = 'yolov8s',
                      pretrained = True):
    # Set Paths
    project_folder = Path('training/yolov8')
    training_run_folder = project_folder / run_name
    training_run_testdata_folder = training_run_folder / 'test'
    dataset_path = Path(dataset_path)
    valset_path = Path(valset_path)
    # Gen training def json
    dataset_def_dict = json.loads(read_textfile(dataset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    valset_def_dict = json.loads(read_textfile(valset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    train_def_dict = {
        'run_name': run_name,
        'disabled_yolo_aug': False,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'model': model,
        'dataset': dataset_def_dict,
        'valset': valset_def_dict,
    }

    print('--- Training...')
    
    model = YOLO(f'{model}.pt')
    model.train(
        data=f'{dataset_path}/{dataset_path.stem}.yaml',
        epochs=epochs, 
        batch=batch_size,
        imgsz=img_size,
        project=project_folder,
        name=run_name,
        pretrained=pretrained,
        exist_ok=True,
        device=list(range(torch.cuda.device_count()))) # Take all GPUs you can find
    
    os.system('rm *.pt')
    
    print('--- Evaluating...')
    gen_evaldata(model, valset_path, create_dir_if_not_exists(training_run_folder / f'test'))
    os.system(f'python evaluation/analyze.py -av {training_run_folder}')
    write_textfile(json.dumps(train_def_dict, indent=4), training_run_testdata_folder / 'training-def.json')

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
        vd_bbox_lines = filter(lambda s: s and s.isspace(), vd_bbox_lines) # Filter whitespace lines away
        
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