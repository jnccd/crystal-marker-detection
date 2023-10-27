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
parser.add_argument('-n','--name', type=str, help='.')
parser.add_argument('-t','--testset-path', type=str, help='.')
parser.add_argument('-df','--dataset-folder', type=str, default='traindata-creator/dataset/_hyp-param-search', help='.') # /data/pcmd/dataset/hyp-param-search
parser.add_argument('-tf','--training-folder', type=str, default='training/hyp-param-search', help='.') # /data/pcmd/training/hyp-param-search
parser.add_argument('-ds','--dataseries-sources', type=str, default='-tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2', help='.')
parser.add_argument('-m','--model', type=str, default='yolov5s', help='Sets the model to train with.')
parser.add_argument('-mine','--min-epochs', type=int, default=5, help='Sets the min epochs to train for in each eval.')
parser.add_argument('-maxe','--max-epochs', type=int, default=15, help='Sets the max epochs to train for in each eval.')
parser.add_argument('-emax','--max-evals', type=int, default=3, help='Sets the max evals to hyp search for.')
parser.add_argument('-us','--use-sahi', action='store_true', help='Use Sahi for inference.')
args = parser.parse_args()

if args.name == None:
    print('I need a name!')
    sys.exit(0)

testset_path = Path(args.testset_path)
dataset_folder = create_dir_if_not_exists(Path(args.dataset_folder) / f'hyp-param-search-{args.name}')

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
history = []

def hyp_param_run(param_dict: dict):
    global best_score, opt_score
    print(f'Hyp params: {param_dict}')
    
    training_folder = create_dir_if_not_exists(Path(args.training_folder) / f'hyp-param-search-{args.name}', clear=True)
    
    dataset_image_size = 0 if args.use_sahi else 640
    dataset_name = 'hyp-search-set'
    dataset_creation_command = f'python traindata-creator/createDataset.py -n {dataset_name} -taf {dataset_folder} -s {dataset_image_size} -sd {param_dict["seed"]} {args.dataseries_sources} -t yolov5 -s {dataset_image_size} -a -aim 4 ' + ' '.join([f'-{x[0]} {param_dict[x[0]]}' for x in def_aug_params])
    print(f'dataset creation command: {dataset_creation_command}')
    os.system(dataset_creation_command)
    
    if yolov5_pattern.match(args.model):
        os.system(f'python batch_train/yolov5.py -d {dataset_folder} -t {testset_path} -e {int(param_dict["epochs"])} -m {args.model} {"-us" if args.use_sahi else ""} -snr -o {training_folder}')
    elif yolov8_pattern.match(args.model):
        os.system(f'python batch_train/yolov8.py -d {dataset_folder} -t {testset_path} -e {int(param_dict["epochs"])} -m {args.model} {"-us" if args.use_sahi else ""} -o {training_folder}')
    elif yolo_nas_pattern.match(args.model):
        os.system(f'python batch_train/yolo_nas.py -d {dataset_folder} -t {testset_path} -e {int(param_dict["epochs"])} -m {args.model} {"-us" if args.use_sahi else ""} -o {training_folder}')
    else:
        print('What model is that? ' + args.model)
        sys.exit(1)
    
    training_subfolder = [x.parent.stem for x in training_folder.glob('**/training-def.json')][0]
    eval_json_path = training_folder / training_subfolder / 'test' / 'evals' / 'evals.json'
    eval_dict = json.loads(read_textfile(eval_json_path).replace("    ", "").replace("\n", ""))
    
    if eval_dict[opt_score] > best_score:
        copy_tree(str(training_folder / training_subfolder), str(training_folder.parent / f'hyp-best-run-{args.name}'))
        best_score = eval_dict[opt_score]
        
    float_param_dict_values = [float(x) for x in param_dict.values()]
    float_param_dict = dict(zip(param_dict.keys(), float_param_dict_values))
    history.append((float(eval_dict[opt_score]), float(best_score), float_param_dict))
    write_textfile(json.dumps(history, indent=4), training_folder.parent / f'hyp-param-search-{args.name}-history.json')
        
    return {'loss': -eval_dict[opt_score], 'status': STATUS_OK }

space = dict([('epochs', hp.uniform('epochs', args.min_epochs, args.max_epochs)), 
              ('seed', hp.randint('seed', 10000))] + 
             [(x[0], hp.uniform(x[0], x[1], x[2])) for x in def_aug_params])

best = fmin(
    hyp_param_run,
    space=space,
    algo=tpe.suggest,
    max_evals=300
    )

print(best)