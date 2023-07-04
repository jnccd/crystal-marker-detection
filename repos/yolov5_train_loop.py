import argparse
import os
from pathlib import Path

root_dir = Path(__file__).resolve().parent
os.chdir(root_dir / '../')
#print(os.getcwd())

parser = argparse.ArgumentParser(prog='yolov5-train-loop', description='Takes a dataset and performs a full training loop with yolov5.')
parser.add_argument('-n','--run-name', type=str, default='default', help='Defines the (folder)name of the run.')
parser.add_argument('-d','--dataset-name', type=str, default='yolov5-640-aug-good-pics-v2', help='Defines the dataset to train from.')
parser.add_argument('-v','--valset-name', type=str, default='yolov5-640-on-skin-valset', help='Defines the valset to test against.')
parser.add_argument('-na','--no-aug', action='store_true', help='Disables the yolov5 repo augmentations.')
parser.add_argument('-s','--img-size', type=int, default=640, help='.')
parser.add_argument('-b','--batch-size', type=int, default=8, help='.')
parser.add_argument('-e','--epochs', type=int, default=100, help='.')
parser.add_argument('-m','--model', type=str, default='yolov5s', help='.')
args = parser.parse_args()

project_folder = Path('repos/training/yolov5')

yolov5_args = ''
if args.no_aug:
    yolov5_args += '--hyp hyp.no-augmentation.yaml'

print('Training...')
os.system(f'python repos/yolov5/train.py --name {args.run_name} --img 640 --batch 8 --epochs 3 --project {project_folder} --data traindata-creator/dataset/{args.dataset_name}/{args.dataset_name}.yaml --weights {args.model}.pt {yolov5_args}')
os.system(f'rm {args.model}.pt')
print('Evaluating...')
os.system(f'python repos/yolov5_gen_evaldata.py -r {args.run_name} -df traindata-creator/dataset/{args.valset_name}/')
os.system(f'python -m cmd_tf -av repos/evaldata/yolov5/{args.run_name}/')
#os.system(f'cat repos/evaldata/yolov5/{args.run_name}/evals/evals.json')