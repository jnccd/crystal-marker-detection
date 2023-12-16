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

from utility import *

parser = argparse.ArgumentParser(prog='find-best-score', description='Finds the best scores achieved in multiple training run folders.')
parser.add_argument('-r','--runs-folders', action='append', nargs='+', type=str, help='Folders which contain the dataset folders in their subfolder tree that should be searched for a good score.')
parser.add_argument('-s','--sort-by', type=str, default='voc2010_mAP', help='The metric to sort by.')
args = parser.parse_args()

root_dir = Path(__file__).resolve().parent
if args.runs_folders is None:
    args.runs_folders = [[root_dir.parent / 'training', root_dir / 'from-server', Path('/data/pcmd/training/')]]
runs_paths = get_all_subfolder_run_dirs(flatten(args.runs_folders))

evals = []
for run_paths_dict in runs_paths:
    eval_dict = json.loads(read_textfile(run_paths_dict['eval']).replace("    ", "").replace("\n", ""))
    train_def_dict = json.loads(read_textfile(run_paths_dict['train_def']).replace("    ", "").replace("\n", ""))
    
    evals.append((run_paths_dict['run_root'], eval_dict, train_def_dict))
    
print(f'Best {args.sort_by} scores:')
evals.sort(key=lambda x: -x[1][args.sort_by])
for i, eval in enumerate(evals[:50]):
    path, dict, train_def_dict = eval
    
    print(str(i+1)+". " + str(dict[args.sort_by]) + ': ' + str(path).replace("\\", "\\\\"))