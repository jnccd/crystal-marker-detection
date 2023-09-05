import argparse
import json
import os
from pathlib import Path
import sys
import time
from distutils.dir_util import copy_tree
from hyperopt import fmin, tpe, hp, STATUS_OK

from evaluation.utility import *

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-t','--testset-path', type=str, help='.')
parser.add_argument('-df','--dataset-folder', type=str, default='traindata-creator/dataset/_hyp-param-search', help='.') # /data/pcmd/dataset/hyp-param-search
parser.add_argument('-tf','--training-folder', type=str, default='training/hyp-param-search', help='.') # /data/pcmd/training/hyp-param-search
parser.add_argument('-ds','--dataseries-sources', type=str, default='-tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2', help='.')
args = parser.parse_args()

testset_path = Path(args.testset_path)
dataset_folder = create_dir_if_not_exists(Path(args.dataset_folder), clear=True)
training_folder = create_dir_if_not_exists(Path(args.training_folder), clear=True)

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
    ("alm2c", 0, 1),
    ("abdc", 0, 1),
    ("alcc", 0, 1),
    ("agnc", 0, 1),
    ("agns", 0, 150),
]

best_score = -1
opt_score = 'voc2010_mAP'

def hyp_param_run(param_dict):
    global best_score, opt_score
    
    training_folder = create_dir_if_not_exists(Path(args.training_folder), clear=True)
    
    dataset_image_size = 640
    dataset_name = 'hyp-search-set'
    os.system(f'python traindata-creator/createDataset.py -n {dataset_name} -taf {dataset_folder} -s {dataset_image_size} -sd {param_dict["seed"]} {args.dataseries_sources} -t yolov5 -s {dataset_image_size} ' +\
        '-a -aim 4 ' + ' '.join([f'-{x[0]} {param_dict[x[0]]}' for x in def_aug_params]))
    
    os.system(f'python batch_train/yolov5.py -d {dataset_folder} -t {testset_path} -e {int(param_dict["epochs"])} -snr -o {training_folder}')
    
    training_subfolder = [x.parent.stem for x in training_folder.glob('**/training-def.json')][0]
    eval_json_path = training_folder / training_subfolder / 'test' / 'evals' / 'evals.json'
    eval_dict = json.loads(read_textfile(eval_json_path).replace("    ", "").replace("\n", ""))
    
    if eval_dict[opt_score] > best_score:
        copy_tree(str(training_folder / training_subfolder), str(training_folder.parent / 'hyp-best-run'))
        best_score = eval_dict[opt_score]
        
    return {'loss': -eval_dict[opt_score], 'status': STATUS_OK }

space = dict([('epochs', hp.uniform('epochs', 5, 15)), 
              ('seed', hp.randint('seed', sys.maxsize))] + 
             [(x[0], hp.uniform(x[0], x[1], x[2])) for x in def_aug_params])

best = fmin(
    hyp_param_run,
    space=space,
    algo=tpe.suggest,
    max_evals=3
    )

print(best)