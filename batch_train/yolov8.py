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
    parser.add_argument('-t','--testset-path', type=str, default='', help='.')
    parser.add_argument('-o','--output-path', type=str, default='training/yolov5', help='.')
    parser.add_argument('-rsf','--recursive-folder-searching', action='store_true', help='.')
    
    parser.add_argument('-s','--img-size', type=int, default=640, help='Sets the img size of the model.')
    parser.add_argument('-b','--batch-size', type=int, default=-1, help='Sets the batch size to train with, -1 is yolov8 AutoBatch.')
    parser.add_argument('-e','--epochs', type=int, default=100, help='Sets the epochs to train for.')
    parser.add_argument('-m','--model', type=str, default='yolov8s', help='Sets the model to train with.')
    parser.add_argument('-rw','--init-random-weights', action='store_true', help='.')
    parser.add_argument('-wn','--weight-noise', type=float, default=0, help='.')
    
    parser.add_argument('-ct','--confidence-threshold', type=float, default=0.5, help='The minimum confidence of considered predictions.')
    parser.add_argument('-bis','--border-ignore-size', type=float, default=0, help='Ignore markers at the border of the image, given in widths from 0 to 0.5.')
    parser.add_argument('-us','--use-sahi', action='store_true', help='Use Sahi for inference.')
    
    parser.add_argument('-wi','--worker-index', type=int, default=-1, help='.')
    parser.add_argument('-wc','--worker-count', type=int, default=-1, help='.')
    
    parser.add_argument('-db','--debug', action='store_true', help='.')
    
    args = parser.parse_args()

    # Paths
    root_dir = Path(__file__).resolve().parent
    datasets_path = Path(args.datasets_path)
    datasets_dirs = [x.parent for x in datasets_path.glob('**/yolov5-*.yaml') 
                    if (not args.recursive_folder_searching and x.parent.parent == datasets_path or args.recursive_folder_searching)
                    and not str(x).__contains__("-valset")]
    datasets_dirs.sort(key=lambda d: d.stem)
    testset_path = Path(args.testset_path)
    
    dd_n = len(datasets_dirs)
    if args.worker_index >= 0 and args.worker_count > 0:
        datasets_dirs = datasets_dirs[int((dd_n / args.worker_count) * args.worker_index):int((dd_n / args.worker_count) * (args.worker_index+1))]
    newline_char = "\n" # Python 3.9 :/
    print(f'Running ensemble run on the following {len(datasets_dirs)} datasets:\n{newline_char.join([str(x) for x in datasets_dirs])}')
    #sys.exit(0) # For dataset choosing testing
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    os.system(f'python traindata-creator/fixYolo5Yamls.py -df {datasets_path}')
    
    # Train
    start_time = time.time()

    loop_folders = datasets_dirs if not args.debug else datasets_dirs[:1]
    for dataset_dir in loop_folders:
        yolov8_train_loop(
            dataset_dir, 
            testset_path, 
            run_name=dataset_dir.stem,
            output_path=args.output_path,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            model_name=args.model,
            weight_noise=args.weight_noise,
            pretrained=not args.init_random_weights,
            use_sahi=args.use_sahi,
            border_ignore_size=args.border_ignore_size,
            confidence_threshold=args.confidence_threshold,
        )
        
    end_time = time.time()
    diff_time = end_time  - start_time
    parsed_time = time.strftime("%H:%M:%S", time.gmtime(diff_time))
    write_textfile(f'{diff_time}\n{parsed_time}', Path(args.output_path) / 'train_time.txt')
    print(f'Training took: {parsed_time}')

def yolov8_train_loop(
    dataset_path, 
    valset_path, 
    output_path = 'training/yolov8',
    run_name = 'default', 
    img_size = 640, 
    batch_size = -1, 
    epochs = 100, 
    model_name = 'yolov8s',
    weight_noise = 0,
    pretrained = True,
    use_sahi = False,
    border_ignore_size = 0,
    confidence_threshold = 0.5,
    ):
    # Set Paths
    project_folder = Path(output_path)
    training_run_folder = project_folder / run_name
    training_run_testdata_folder = training_run_folder / 'test'
    dataset_path = Path(dataset_path)
    valset_path = Path(valset_path)
    print('Training in: ', dataset_path)
    # Gen training def json
    dataset_def_dict = json.loads(read_textfile(dataset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    valset_def_dict = json.loads(read_textfile(valset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    train_def_dict = {
        'run_name': run_name,
        'disabled_yolo_aug': False,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'model': model_name,
        'dataset': dataset_def_dict,
        'valset': valset_def_dict,
    }

    print('--- Training...')
    
    model = YOLO(f'{model_name}.pt')
    # Add noise to model if arg is set
    if weight_noise > 0:
        with torch.no_grad():
            for param in model.model.parameters():
                param.add_(torch.randn(param.size()) * weight_noise)
    model.train(
        data=f'{dataset_path}/{dataset_path.stem}.yaml',
        epochs=epochs, 
        batch=batch_size,
        imgsz=img_size,
        project=project_folder,
        name=run_name,
        pretrained=pretrained,
        exist_ok=True,
        device=[0])#list(range(torch.cuda.device_count())))
    
    os.system('rm *.pt')
    
    print('--- Evaluating...')
    os.system(f'python batch_train/yolov8_evaluate.py -r {training_run_folder} -t {valset_path} {"-us" if use_sahi else ""} -bis {border_ignore_size} -ct {confidence_threshold}')
    write_textfile(json.dumps(train_def_dict, indent=4), training_run_folder / 'training-def.json')

if __name__ == '__main__':
    main()