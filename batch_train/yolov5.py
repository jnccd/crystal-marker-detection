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
    parser.add_argument('-n','--name', type=str, default='yolov5', help='.')
    
    parser.add_argument('-s','--img-size', type=int, default=640, help='Sets the img size of the model.')
    parser.add_argument('-b','--batch-size', type=int, default=8, help='Sets the batch size to train with.')
    parser.add_argument('-e','--epochs', type=int, default=100, help='Sets the epochs to train for.')
    parser.add_argument('-m','--model', type=str, default='yolov5s', help='Sets the model to train with.')
    parser.add_argument('-rw','--init-random-weights', action='store_true', help='.')
    
    parser.add_argument('-wi','--worker-index', type=int, default=-1, help='.')
    parser.add_argument('-wc','--worker-count', type=int, default=-1, help='.')
    
    parser.add_argument('-db','--debug', action='store_true', help='.')
    
    args = parser.parse_args()

    # Paths
    root_dir = Path(__file__).resolve().parent
    datasets_path = root_dir.parent / args.dataset_path
    datasets_dirs = [x.parent for x in datasets_path.glob('**/yolov5-*.yaml') 
                    if x.parent.parent == datasets_path
                    and not str(x).__contains__("-valset")]
    testset_path = root_dir.parent / args.testset_path
    
    dd_n = len(datasets_dirs)
    if args.worker_index >= 0 and args.worker_count > 0:
        datasets_dirs = datasets_dirs[int((dd_n / args.worker_count) * args.worker_index):int((dd_n / args.worker_count) * (args.worker_index+1))]
    newline_char = "\n" # Python 3.9 :/
    print(f'Running ensample run on the following {len(datasets_dirs)} datasets:\n{newline_char.join([str(x) for x in datasets_dirs])}')
    
    os.system(f'python traindata-creator/fixYolo5Yamls.py -df {datasets_path}')
    
    # Train
    start_time = time.time()

    loop_folders = datasets_dirs if not args.debug else datasets_dirs[:1]
    for dataset_dir in loop_folders:
        # Without yolov5 aug
        yolov5_train_loop(dataset_dir, 
                          testset_path, 
                          run_name=dataset_dir.stem,
                          ensample_name=args.name,
                          epochs=args.epochs,
                          img_size=args.img_size,
                          batch_size=args.batch_size,
                          model=args.model,
                          init_random_weights=args.init_random_weights,
                          no_aug=True)
        # With yolov5 aug
        yolov5_train_loop(dataset_dir, 
                          testset_path, 
                          run_name=dataset_dir.stem+'-yolo5aug',
                          ensample_name=args.name,
                          epochs=args.epochs,
                          img_size=args.img_size,
                          batch_size=args.batch_size,
                          model=args.model,
                          init_random_weights=args.init_random_weights,
                          no_aug=False)
        
    end_time = time.time()
    diff_time = end_time  - start_time
    print(f'Training took: {time.strftime("%H:%M:%S", time.gmtime(diff_time))}')

def yolov5_train_loop(dataset_path, 
                      valset_path, 
                      ensample_name = 'yolov5',
                      run_name = 'default', 
                      img_size = 640, 
                      batch_size = 8, 
                      epochs = 100, 
                      model = 'yolov5s', 
                      init_random_weights = False, 
                      no_aug = False):
    # --- Set Paths
    project_folder = Path('training') / ensample_name
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
    os.system(f'python repos/yolov5_gen_evaldata.py -r {training_run_folder} -df {valset_path}/')
    os.system(f'python evaluation/analyze.py -av {training_run_folder}')
    write_textfile(json.dumps(train_def_dict, indent=4), training_run_testdata_folder / 'training-def.json')

if __name__ == '__main__':
    main()