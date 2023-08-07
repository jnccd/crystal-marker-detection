import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path

from utility import *

# Parse
parser = argparse.ArgumentParser(prog='', description='Regenerate evaldata and reanalyze outputs, in case there was a bug in the prod server script.')
parser.add_argument('-r','--runs-path', type=str, help='.')
parser.add_argument('-t','--testset-path', type=str, help='.')
parser.add_argument('-rt','--run-type', type=str, help='.')

parser.add_argument('-rne','--run-name-exclude', type=str, default='', help='.')
args = parser.parse_args()

run_dirs = [(x.parent.parent, x) for x in Path(args.runs_path).glob('**/training-def.json')]
newline_char = "\n" # Python 3.9 :/
print(f'Running reeval run on the following {len(run_dirs)} training runs:\n{newline_char.join([str(x[0]) for x in run_dirs])}')

yolov5_pattern = re.compile('yolov5(.?)$')
yolov8_pattern = re.compile('yolov8(.?)$|yolov5(.?)u$') 

for training_run_folder, train_def_path in run_dirs:
    if str(training_run_folder).__contains__(args.run_name_exclude):
        print(f'Skipping {training_run_folder}...')
        continue
    
    # Read train def
    train_def_json = json.loads(read_textfile(train_def_path).replace("    ", "").replace("\n", ""))
    train_def_model = train_def_json['model']
    
    # Set run_type
    if args.run_type is not None:
        current_run_type = args.run_type
    else:
        if yolov5_pattern.match(train_def_model):
            current_run_type = 'yolov5'
        elif yolov8_pattern.match(train_def_model):
            current_run_type = 'yolov8'
        else:
            print('What model is that? ' + train_def_model)
            sys.exit(1)
            
    # Regenerate
    print(current_run_type)
    os.system(f'rm {training_run_folder / "test" / "*.txt"}'.replace("\\", "/"))
    os.system(f'rm {training_run_folder / "test" / "*.png"}'.replace("\\", "/"))
    if current_run_type == 'yolov5':
        os.system(f'python repos/yolov5_gen_evaldata.py -r {training_run_folder} -df {args.testset_path}/')
        os.system(f'python evaluation/analyze.py -av {training_run_folder}')
    elif current_run_type == 'yolov8':
        os.system(f'python batch_train/yolov8_gen_evaldata.py -r {training_run_folder} -df {args.testset_path}/')
        os.system(f'python evaluation/analyze.py -av {training_run_folder}')
