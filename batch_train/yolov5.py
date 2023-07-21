import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

from utils import *

def main():
    # Parse
    parser = argparse.ArgumentParser(prog='', description='.')
    parser.add_argument('-d','--dataset-path', type=str, default='', help='.')
    parser.add_argument('-v','--testset-path', type=str, default='', help='.')
    args = parser.parse_args()

    # Paths
    root_dir = Path(__file__).resolve().parent
    dataset_path = root_dir.parent / args.dataset_path
    dataset_dirs = [x for x in dataset_path.glob('**/yolov5-*/') 
                    if x.is_dir() 
                        and not str(x).__contains__("_old") 
                        and not str(x).__contains__("-valset")]
    testset_path = root_dir.parent / args.testset_path
    newline_char = "\n" # Python 3.9 :/
    print(f'Running ensample run using the following datasets:\n{newline_char.join([str(x) for x in dataset_dirs])}')
    
    # Train
    start_time = time.time()

    for dataset_dir in dataset_dirs[:1]:
        # Without yolov5 aug
        yolov5_train_loop(dataset_dir, 
                          testset_path, 
                          run_name=dataset_dir.stem,
                          epochs=2,
                          model='yolov5s',
                          init_random_weights=True,
                          no_aug=True)
        # With yolov5 aug
        yolov5_train_loop(dataset_dir, 
                          testset_path, 
                          run_name=dataset_dir.stem+'-yolo5aug',
                          epochs=2,
                          model='yolov5s',
                          init_random_weights=True,
                          no_aug=False)
        
    end_time = time.time()
    diff_time = end_time  - start_time
    print(f'Training took: {time.strftime("%H:%M:%S", time.gmtime(diff_time))}')

def yolov5_train_loop(dataset_path, 
                      valset_path, 
                      run_name = 'default', 
                      img_size = 640, 
                      batch_size = 8, 
                      epochs = 100, 
                      model = 'yolov5s', 
                      init_random_weights = False, 
                      no_aug = False):
    # --- Set Paths
    project_folder = Path('training/yolov5')
    training_run_folder = project_folder / run_name
    training_run_testdata_folder = training_run_folder / 'test'
    dataset_path = Path(dataset_path)
    valset_path = Path(valset_path)
    # --- Gen training def json
    dataset_def_dict = json.loads(read_textfile(dataset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    valset_def_dict = json.loads(read_textfile(valset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    train_def_dict = {
        'run_name': run_name,
        'disabled_yolo_aug': no_aug,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'model': model,
        'dataset': dataset_def_dict,
        'valset': valset_def_dict,
    }

    # --- Set extra arguments
    yolov5_args = ''

    # Random yolov5 weight init
    if init_random_weights:
        yolov5_args += f"--weights '' --cfg {model}.yaml "
    else:
        yolov5_args += f'--weights {model}.pt '

    # Disable yolov5 augmentation
    if no_aug:
        yolov5_args += '--hyp hyp.no-augmentation.yaml '
        
    # --- Commands
    print('--- Training...')
    os.system(f'python repos/yolov5/train.py --name {run_name} --img {img_size} --batch {batch_size} --epochs {epochs} --project {project_folder} --data {dataset_path}/{dataset_path.stem}.yaml {yolov5_args}')
    os.system(f'rm {model}.pt')
    print('--- Evaluating...')
    os.system(f'python repos/yolov5_gen_evaldata.py -r {run_name} -df {valset_path}/')
    os.system(f'python evaluation/analyze.py -av {training_run_folder}')
    write_textfile(json.dumps(train_def_dict, indent=4), training_run_testdata_folder / 'training-def.json')

if __name__ == '__main__':
    main()