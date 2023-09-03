import argparse
import json
import os
from pathlib import Path
import sys
import time
from hyperopt import fmin, tpe, hp, STATUS_OK

from evaluation.utility import read_textfile

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-t','--testset-path', type=str, help='.')
parser.add_argument('-df','--dataset-folder', type=str, default='traindata-creator/dataset', help='.') # /data/pcmd/dataset/
parser.add_argument('-tf','--training-folder', type=str, default='training/', help='.') # /data/pcmd/training/
args = parser.parse_args()

testset_path = Path(args.testset_path)
dataset_folder = Path(args.dataset_folder)
training_folder = Path(args.training_folder)

def_aug_params = [
    ("asgsc", 0, 1),
    ("apldc", 0, 1),
    ("apc", 0, 1),
    ("aps", 0, 0.5),
    ("arc", 0, 1),
    ("ars", 0, 360),
    ("andrc", 0, 1),
    ("arcc", 0, 1),
    ("arc2c", 0, 1),
    ("almc", 0, 1),
    ("abdc", 0, 1),
    ("alcc", 0, 1),
    ("agnc", 0, 1),
    ("agns", 0, 150),
]

def hyp_param_run(param_dict):
    
    dataset_image_size = 0
    dataset_name = 'hyp-search-set'
    os.system(f'python traindata-creator/createDataset.py -n {dataset_name} -taf {dataset_folder} -s {dataset_image_size} -sd {param_dict["seed"]} ' +\
        '-a -aim 4 ' + ' '.join([f'-{x[0]} {param_dict[x[0]]}' for x in def_aug_params]))
    
    os.system(f'python batch_train/yolov5.py -d {dataset_folder / dataset_name} -t {testset_path} -e {param_dict["epochs"]} -snr -o {training_folder}')
    
    eval_json_path = training_folder / f'yolov5-{dataset_image_size}-{dataset_name}' / 'test' / 'evals' / 'evals.json'
    eval_dict = json.loads(read_textfile(eval_json_path).replace("    ", "").replace("\n", ""))
    
    return {'loss': -eval_dict['voc2010_mAP'], 'status': STATUS_OK }

space = dict([('epochs', hp.uniform('epochs', 25, 400)), 
              ('seed', hp.randint('seed', sys.maxsize))] + 
             [(x[0], hp.uniform(x[0], x[1], x[2])) for x in def_aug_params])

best = fmin(
    hyp_param_run,
    space=space,
    algo=tpe.suggest,
    max_evals=3
    )

print(best)