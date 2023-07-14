import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

start_time = time.time()

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-d','--dataset-path', type=str, default='', help='.')
parser.add_argument('-v','--valset-path', type=str, default='', help='.')
args = parser.parse_args()

root_dir = Path(__file__).resolve().parent
dataset_path = root_dir.parent / args.dataset_path
dataset_dirs = [x for x in dataset_path.glob('**/yolov5-*/') 
                if x.is_dir() 
                    and not str(x).__contains__("_old") 
                    and not str(x).__contains__("-valset")]
valset_path = root_dir.parent / args.valset_path
newline_char = "\n" # Python 3.9 :/
print(f'Running ensample run using the following datasets:\n{newline_char.join([str(x) for x in dataset_dirs])}')

for dir in dataset_dirs[:1]:
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-1 --no-aug -d {dir} -v {valset_path} -e 300')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-2 --no-aug -d {dir} -v {valset_path} -e 300')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-3 --no-aug -d {dir} -v {valset_path} -e 300')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-yolo5aug-1 -d {dir} -v {valset_path} -e 300')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-yolo5aug-2 -d {dir} -v {valset_path} -e 300')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-yolo5aug-3 -d {dir} -v {valset_path} -e 300')
    
end_time = time.time()
diff_time = end_time  - start_time
print(f'Training took: {time.strftime("%H:%M:%S", time.gmtime(diff_time))}')