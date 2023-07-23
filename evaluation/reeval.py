import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

from utils import *

# Parse
parser = argparse.ArgumentParser(prog='', description='Regenerate evaldata and reanalyze outputs, in case there was a bug in the prod server script.')
parser.add_argument('-r','--runs-path', type=str, default='', help='.')
parser.add_argument('-v','--testset-path', type=str, default='', help='.')
parser.add_argument('-t','--run-type', type=str, default='yolov5', help='.')
args = parser.parse_args()

run_dirs = [x.parent.parent for x in Path(args.runs_path).glob('**/training-def.json')]

for training_run_folder in run_dirs:
    if args.run_type == 'yolov5':
        os.system(f'python repos/yolov5_gen_evaldata.py -r {training_run_folder} -df {args.testset_path}/')
        os.system(f'python evaluation/analyze.py -av {training_run_folder}')
    elif args.run_type == 'yolov8':
        os.system(f'python batch_train/yolov8_gen_evaldata.py -r {training_run_folder} -df {args.testset_path}/')
        os.system(f'python evaluation/analyze.py -av {training_run_folder}')