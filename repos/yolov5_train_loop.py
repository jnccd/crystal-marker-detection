import argparse
import ast
import json
import os
from pathlib import Path
from utils import *

root_dir = Path(__file__).resolve().parent
os.chdir(root_dir / '../')
#print(os.getcwd())

parser = argparse.ArgumentParser(prog='yolov5-train-loop', description='Takes a dataset and performs a full training loop with yolov5.')
parser.add_argument('-n','--run-name', type=str, default='default', help='Defines the (folder)name of the run.')
parser.add_argument('-d','--dataset-path', type=str, default='traindata-creator/dataset/yolov5-640-aug-good-pics-v2', help='Defines the dataset to train from.')
parser.add_argument('-v','--valset-path', type=str, default='traindata-creator/dataset/yolov5-640-on-skin-valset', help='Defines the valset to test against.')
parser.add_argument('-na','--no-aug', action='store_true', help='Disables the yolov5 repo augmentations.')
parser.add_argument('-s','--img-size', type=int, default=640, help='Sets the img size of the model.')
parser.add_argument('-b','--batch-size', type=int, default=8, help='Sets the batch size to train with.')
parser.add_argument('-e','--epochs', type=int, default=100, help='Sets the epochs to train for.')
parser.add_argument('-m','--model', type=str, default='yolov5s', help='Sets the model to train with.')
args = parser.parse_args()

project_folder = Path('repos/training/yolov5')
dataset_path = Path(args.dataset_path)
valset_path = Path(args.valset_path)
dataset_def_dict = json.loads(read_textfile(dataset_path / 'dataset-def.json').replace(" ", "").replace("\n", ""))
valset_def_dict = json.loads(read_textfile(dataset_path / 'dataset-def.json').replace(" ", "").replace("\n", ""))
train_def_dict = {
    'run_name': args.run_name,
    'disabled_yolo_aug': args.no_aug,
    'img_size': args.img_size,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'model': args.model,
    'valset': valset_def_dict,
    'dataset': dataset_def_dict,
}
evaldata_run_dir = Path(f'repos/evaldata/yolov5/{args.run_name}/')

yolov5_args = ''
if args.no_aug:
    yolov5_args += '--hyp hyp.no-augmentation.yaml'

print('--- Training...')
os.system(f'python repos/yolov5/train.py --name {args.run_name} --img {args.img_size} --batch {args.batch_size} --epochs {args.epochs} --project {project_folder} --data {dataset_path}/{dataset_path.stem}.yaml --weights {args.model}.pt {yolov5_args}')
os.system(f'rm {args.model}.pt')
print('--- Evaluating...')
os.system(f'python repos/yolov5_gen_evaldata.py -r {args.run_name} -df {args.valset_path}/')
os.system(f'python evaluation/analyze.py -av {evaldata_run_dir}')
write_textfile(json.dumps(train_def_dict, indent=4), evaldata_run_dir / 'training-def.json')