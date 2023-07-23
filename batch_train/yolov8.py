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
    parser.add_argument('-n','--name', type=str, default='yolov8', help='.')
    
    parser.add_argument('-s','--img-size', type=int, default=640, help='Sets the img size of the model.')
    parser.add_argument('-b','--batch-size', type=int, default=-1, help='Sets the batch size to train with, -1 is yolov8 AutoBatch.')
    parser.add_argument('-e','--epochs', type=int, default=100, help='Sets the epochs to train for.')
    parser.add_argument('-m','--model', type=str, default='yolov8s', help='Sets the model to train with.')
    parser.add_argument('-rw','--init-random-weights', action='store_true', help='.')
    
    args = parser.parse_args()

    # Paths
    root_dir = Path(__file__).resolve().parent
    datasets_path = Path(args.datasets_path)
    datasets_dirs = [x.parent for x in datasets_path.glob('**/yolov5-*.yaml') 
                    if not str(x).__contains__("_old") 
                    and not str(x).__contains__("-valset")]
    testset_path = Path(args.testset_path)
    newline_char = "\n" # Python 3.9 :/
    print(f'Running ensample run using the following datasets:\n{newline_char.join([str(x) for x in datasets_dirs])}')
    
    # Train
    start_time = time.time()

    for dataset_dir in datasets_dirs:
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
                      ensample_name = 'yolov8',
                      run_name = 'default', 
                      img_size = 640, 
                      batch_size = -1, 
                      epochs = 100, 
                      model = 'yolov8s',
                      pretrained = True):
    # Set Paths
    project_folder = Path('training') / ensample_name
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
        device=[0])#list(range(torch.cuda.device_count()))) # Take all GPUs you can find
    
    os.system('rm *.pt')
    
    print('--- Evaluating...')
    os.system(f'python batch_train/yolov8_gen_evaldata.py -r {training_run_folder} -df {valset_path}')
    os.system(f'python evaluation/analyze.py -av {training_run_folder}')
    write_textfile(json.dumps(train_def_dict, indent=4), training_run_testdata_folder / 'training-def.json')

if __name__ == '__main__':
    main()