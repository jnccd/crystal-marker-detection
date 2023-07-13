import argparse
import ast
import json
import os
from pathlib import Path
from utils import *

parser = argparse.ArgumentParser(prog='fix-yolov5', description='Fixes yolov5 yaml paths after dataset folder has moved.')
parser.add_argument('-df','--dataset-path', type=str,  help='')
args = parser.parse_args()

dataset_path = Path(args.dataset_path)
for yaml_path in dataset_path.rglob("yolov5-*.yaml"):
    lines = read_textfile(yaml_path).split('\n')
    lines[0] = f"path: {yaml_path.parent.resolve()}"
    write_textfile('\n'.join(lines), yaml_path)