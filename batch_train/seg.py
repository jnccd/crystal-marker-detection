import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

from utils import *

# Segmentation models ensample run

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(prog='', description='.')
    parser.add_argument('-d','--dataset-path', type=str, default='', help='.')
    parser.add_argument('-v','--testset-path', type=str, default='', help='.')
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    dataset_path = root_dir.parent / args.dataset_path
    dataset_dirs = [x for x in dataset_path.glob('**/seg-*/') 
                    if x.is_dir() 
                        and not str(x).__contains__("_old") 
                        and not str(x).__contains__("-valset")]
    testset_path = root_dir.parent / args.testset_path
    newline_char = "\n" # Python 3.9 :/
    print(f'Running ensample run using the following datasets:\n{newline_char.join([str(x) for x in dataset_dirs])}')

    # Main train loop
    for dir in dataset_dirs:
        segmentation_model_train_loop('sm-unet', dir, testset_path, epochs=2, img_size=320)
        # TODO: Add more training runs
        
    end_time = time.time()
    diff_time = end_time  - start_time
    print(f'Training took: {time.strftime("%H:%M:%S", time.gmtime(diff_time))}')
    
def segmentation_model_train_loop(run_name_prefix, dataset_path, testset_path, epochs = 100, batch_size = 8, img_size = 640):
    run_name = f'{run_name_prefix}-{dataset_path.stem}'
    
    os.system(f'python -m cmd_tf -df {dataset_path} -r {run_name} -bs {batch_size} -e {epochs} -s {img_size}')
    os.system(f'python -m cmd_tf -t -td {testset_path} -r {run_name}')
    os.system(f'python evaluation/analyze.py -av {run_name}')
    
    dataset_def_dict = json.loads(read_textfile(dataset_path / 'dataset-def.json').replace(" ", "").replace("\n", ""))
    valset_def_dict = json.loads(read_textfile(testset_path / 'dataset-def.json').replace(" ", "").replace("\n", ""))
    train_def_dict = {
        'run_name': run_name,
        'run_name_prefix': run_name_prefix,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'valset': valset_def_dict,
        'dataset': dataset_def_dict,
    }
    write_textfile(json.dumps(train_def_dict, indent=4), Path(f'cmd_tf/runs/{run_name}/test') / 'training-def.json')
    
if __name__ == '__main__':
    main()