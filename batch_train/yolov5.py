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
parser.add_argument('-v','--testset-path', type=str, default='', help='.')
args = parser.parse_args()

root_dir = Path(__file__).resolve().parent
dataset_path = root_dir.parent / args.dataset_path
dataset_dirs = [x for x in dataset_path.glob('**/yolov5-*/') 
                if x.is_dir() 
                    and not str(x).__contains__("_old") 
                    and not str(x).__contains__("-valset")]
testset_path = root_dir.parent / args.testset_path
newline_char = "\n" # Python 3.9 :/
print(f'Running ensample run using the following datasets:\n{newline_char.join([str(x) for x in dataset_dirs])}')

for dir in dataset_dirs:
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem} --no-aug -d {dir} -v {testset_path} -e 300 -m yolov5m -rw')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-yolo5aug -d {dir} -v {testset_path} -e 300 -m yolov5m -rw')
    
end_time = time.time()
diff_time = end_time  - start_time
print(f'Training took: {time.strftime("%H:%M:%S", time.gmtime(diff_time))}')