import argparse
import json
import os
import math
import random
from pathlib import Path
import shutil
import sys
import re

import numpy as np

from utils import *

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-rf','--runs-folders', action='append', nargs='+', type=str, help='.')
parser.add_argument('-s','--sort-by', type=str, default='voc2010_mAP', help='.')
args = parser.parse_args()

root_dir = Path(__file__).resolve().parent
if args.runs_folders is None:
    args.runs_folders = [[root_dir / 'from-server', root_dir.parent / 'training']]
runs_folders = flatten(args.runs_folders)
eval_paths = flatten([[ x for x in Path(run_folders).glob('**/evals.json') 
                        if not str(x).__contains__("_old")] 
                        for run_folders in runs_folders])

evals = []
for eval_path in eval_paths:
    train_def_path = eval_path.parent.parent / 'training-def.json'
    
    if not train_def_path.is_file():
            print(f'Couldnt find train file of {eval_path}')
            continue
    
    eval_dict = json.loads(read_textfile(eval_path).replace("    ", "").replace("\n", ""))
    train_def_dict = json.loads(read_textfile(train_def_path).replace("    ", "").replace("\n", ""))
    
    evals.append((eval_path, eval_dict, train_def_dict))
    
evals.sort(key=lambda x: -x[1][args.sort_by])

for eval in evals[:10]:
    path, dict, train_def_dict = eval
    
    print(f'{dict[args.sort_by]}: {path}')