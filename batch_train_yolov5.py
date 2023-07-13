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
args = parser.parse_args()

dataset_dirs = [x for x in Path(args.dataset_path).glob('**/yolov5-*/') 
                if x.is_dir() 
                    and not str(x).__contains__("_old") 
                    and not str(x).__contains__("-valset")]
#print(dataset_dirs)

for dir in dataset_dirs[:1]:
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-1 -d {dir} -e 300 --no-aug')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-2 -d {dir} -e 300 --no-aug')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-3 -d {dir} -e 300 --no-aug')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-yolo5aug-1 -d {dir} -e 300')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-yolo5aug-2 -d {dir} -e 300')
    os.system(f'python repos/yolov5_train_loop.py -n {dir.stem}-yolo5aug-3 -d {dir} -e 300')
    
end_time = time.time()
diff_time = end_time  - start_time
print(f'Training took: {time.strftime("%H:%M:%S", time.gmtime(diff_time))}')