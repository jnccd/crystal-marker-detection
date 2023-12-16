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
parser = argparse.ArgumentParser(prog='reeval', description='Regenerate evaldata and reanalyze outputs, in case there was a bug in the prod server script.')
parser.add_argument('-r','--runs-folders', action='append', nargs='+', type=str, help='Folders which contain the dataset folders in their subfolder tree that should be reevaluated.')
parser.add_argument('-t','--testset-path', type=str, help='The dataset to use the valdata of as a testset for this evaluation.')
parser.add_argument('-rt','--run-type', type=str, help='The model type of the runs, like yolov5 or yolov8. This can also be inferred automatically using the training metadata')

parser.add_argument('-rne','--run-name-exclude', type=str, default='---------', help='Regex for run names to exclude from reeval.')
parser.add_argument('-sge','--skip-gen-evaldata', action='store_true', help='Only calls analyze.')
parser.add_argument('-stn','--set-test-folder-name', action='store_true', help='Sets the test folder name in the run folder based on the settings.')

parser.add_argument('-ct','--confidence-threshold', type=float, default=0.5, help='The minimum confidence of considered predictions.')
parser.add_argument('-bis','--border-ignore-size', type=float, default=0, help='Ignore markers at the border of the image, given in widths from 0 to 0.5.')
parser.add_argument('-sqt','--squareness-threshold', type=float, default=0, help='The minimum squareness of considered prediction boxes.')
parser.add_argument('-us','--use-sahi', action='store_true', help='Use Sahi for inference.')
args = parser.parse_args()

runs_paths = get_all_subfolder_run_dirs(flatten(args.runs_folders))
newline_char = "\n" # Python 3.9 :/
print(f'Running reeval run on the following {len(runs_paths)} training runs:\n{newline_char.join([str(x["run_root"]) for x in runs_paths])}')

for run_paths_dict in runs_paths:
    training_run_folder = run_paths_dict['run_root']
    train_def_path = run_paths_dict['train_def']
    if str(training_run_folder).__contains__(args.run_name_exclude):
        print(f'Skipping {training_run_folder}...')
        continue
    
    legacy_train_def_location = training_run_folder / 'test/training-def.json'
    if legacy_train_def_location.is_file():
        # In case of old train def config, save the file from deletion by the evaluation scripts by copying it
        print(f'Saving train def to {training_run_folder / "training-def.json"}...')
        shutil.copyfile(legacy_train_def_location, training_run_folder / 'training-def.json')
    
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
    
    # Set test folder name
    testset_path = Path(args.testset_path)
    if args.set_test_folder_name:
        test_folder_name = f'test-{testset_path.stem}{"-sahi" if args.use_sahi else ""}{f"-ct{round(args.confidence_threshold * 100)}" if args.confidence_threshold != 0.5 else ""}' + \
                           f'{f"-bis{round(args.border_ignore_size * 100)}" if args.border_ignore_size != 0 else ""}{f"-sqt{round(args.squareness_threshold * 100)}" if args.squareness_threshold != 0 else ""}'
    else:
        test_folder_name = 'test'
    
    # Regenerate
    print("Run type:", current_run_type)
    if args.skip_gen_evaldata:
        os.system(f'python evaluation/analyze.py -av {training_run_folder / test_folder_name}')
    else:
        if current_run_type == 'yolov5':
            os.system(f'python repos/yolov5_evaluate.py -r {training_run_folder} -t {testset_path}/ -ct {args.confidence_threshold} -bis {args.border_ignore_size} -sqt {args.squareness_threshold} -tn {test_folder_name}')
        elif current_run_type == 'yolov8':
            os.system(f'python batch_train/yolov8_evaluate.py -r {training_run_folder} -t {testset_path}/')
