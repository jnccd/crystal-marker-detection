import argparse
import os
import random
import shutil
import sys
import time
import cv2
from pathlib import Path

from utils import *

train_dir_name = 'train'
val_dir_name = 'val'
dataset_name = ''
dataset_dir = None

def main():
    global train_dir_name, val_dir_name, dataset_name, dataset_dir
    
    parser = argparse.ArgumentParser(prog='dataset-creator', description='Combines multiple dataseries to a dataset.')
    parser.add_argument('-n','--name', type=str, help='Defines the (folder)name of the dataset.')
    parser.add_argument('-t','--type', type=str, help='Defines the type of dataset to be build, "seg" for segmentation, "yolov5" for yolov5 od (object detection), "csv" for csv od.')
    parser.add_argument('-tf','--traindata-folders', action='append', nargs='+', type=str, help='The folders containing train data.')
    parser.add_argument('-vf','--valdata-folders', action='append', nargs='+', type=str, help='The folders containing validation data.')
    parser.add_argument('-r','--ratio', type=float, help='Ratio of traindata to be assigned to valdata, if set overrides the -vf setting.')
    args = parser.parse_args()
    
    dataset_name = f'{args.type}-{args.name}'
    print(f'Creating {dataset_name}...')
    
    # --- Get Folders ---
    td_folders = flatten(args.traindata_folders)
    
    if args.ratio is None:
        vd_folders = flatten(args.valdata_folders)
        
        td_in_paths = get_files_from_folders_with_ending(td_folders, '_in.png')
        vd_in_paths = get_files_from_folders_with_ending(vd_folders, '_in.png')
    else:
        full_td_in_paths = get_files_from_folders_with_ending(td_folders, '_in.png')
        random.Random(42).shuffle(full_td_in_paths)
        
        n = len(full_td_in_paths)
        nt = int(n * (1-args.ratio))
        
        td_in_paths = full_td_in_paths[:nt]
        vd_in_paths = full_td_in_paths[nt:]
        #print(len(td_in_paths))
        #print(len(vd_in_paths))
    
    root_dir = Path(__file__).resolve().parent
    dataset_dir = create_dir_if_not_exists(root_dir / 'dataset' / dataset_name, clear=True)
    
    # --- Build dataset ---
    if args.type == 'seg':
        build_seg_dataset(td_in_paths, vd_in_paths)
    elif args.type == 'yolov5':
        build_yolov5_dataset(td_in_paths, vd_in_paths)
    elif args.type == 'csv':
        build_od_csv_dataset(td_in_paths, vd_in_paths)
    else:
        print('Error: Unsupported dataset type!')
   
def build_seg_dataset(td_in_paths, vd_in_paths):
    global train_dir_name, val_dir_name, dataset_name, dataset_dir
    
    train_dir = create_dir_if_not_exists(dataset_dir / train_dir_name)
    val_dir = create_dir_if_not_exists(dataset_dir / val_dir_name)
    
    td_seg_paths = get_adjacet_files_with_ending(td_in_paths, '_seg.png')
    vd_seg_paths = get_adjacet_files_with_ending(vd_in_paths, '_seg.png')
    
    for i, (td_in, td_seg) in enumerate(zip(td_in_paths, td_seg_paths)):
        shutil.copyfile(td_in, train_dir / f'{i}_in.png')
        shutil.copyfile(td_seg, train_dir / f'{i}_seg.png')
    print(f'Built {i+1} traindata!')
    
    for i, (vd_in, vd_seg) in enumerate(zip(vd_in_paths, vd_seg_paths)):
        shutil.copyfile(vd_in, val_dir / f'{i}_in.png')
        shutil.copyfile(vd_seg, val_dir / f'{i}_seg.png')
    print(f'Built {i+1} valdata!')
    
    
def build_od_csv_dataset(td_in_paths, vd_in_paths):
    global train_dir_name, val_dir_name, dataset_name, dataset_dir
    
    # Create train / val dirs
    train_dir = create_dir_if_not_exists(dataset_dir / train_dir_name)
    val_dir = create_dir_if_not_exists(dataset_dir / val_dir_name)
    
    # Get BBox files
    td_bbox_paths = get_adjacet_files_with_ending(td_in_paths, '_xyxy.txt')
    vd_bbox_paths = get_adjacet_files_with_ending(vd_in_paths, '_xyxy.txt')
    
    # Build traindata
    traindata_csv_entries = []
    traindata_csv_path = dataset_dir / 'train.csv'
    for i, (td_in, td_bbox) in enumerate(zip(td_in_paths, td_bbox_paths)):
        pic_filename = f'{i}.png'
        pic_path = train_dir / pic_filename
        shutil.copyfile(td_in, pic_path)
        with open(td_bbox, 'r') as file:
            td_bbox_lines = file.read().split('\n')
        td_bbox_lines.pop()
        for line in td_bbox_lines:
            traindata_csv_entries.append(f'{str(train_dir / pic_filename)},{",".join([x.split(".")[0] for x in line.split(" ")])},marker') # Low Prio TODO: Add more classes
    with open(traindata_csv_path, "w") as text_file:
        for entry in traindata_csv_entries:
            text_file.write(f"{entry}\n")
    print(f'Built {i} traindata!')
    
    # Build valdata
    valdata_csv_entries = []
    valdata_csv_path = dataset_dir / 'val.csv'
    for i, (vd_in, vd_bbox) in enumerate(zip(vd_in_paths, vd_bbox_paths)):
        pic_filename = f'{i}.png'
        pic_path = val_dir / pic_filename
        shutil.copyfile(vd_in, pic_path)
        with open(vd_bbox, 'r') as file:
            vd_bbox_lines = file.read().split('\n')
        vd_bbox_lines.pop()
        for line in vd_bbox_lines:
            valdata_csv_entries.append(f'{str(val_dir / pic_filename)},{",".join([x.split(".")[0] for x in line.split(" ")])},marker') # Low Prio TODO: Add more classes
    with open(valdata_csv_path, "w") as text_file:
        for entry in valdata_csv_entries:
            text_file.write(f"{entry}\n")
    print(f'Built {i} valdata!')
    
    classes_csv_path = dataset_dir / 'classes.csv'
    with open(classes_csv_path, "w") as text_file:
        text_file.write(f"marker,0\n")
    
def build_yolov5_dataset(td_in_paths, vd_in_paths):
    global train_dir_name, val_dir_name, dataset_name, dataset_dir
    
    # Create train / val dirs
    train_dir = create_dir_if_not_exists(dataset_dir / train_dir_name)
    val_dir = create_dir_if_not_exists(dataset_dir / val_dir_name)
    
    # Get BBox files
    td_bbox_paths = get_adjacet_files_with_ending(td_in_paths, '_cxcywh_n.txt')
    vd_bbox_paths = get_adjacet_files_with_ending(vd_in_paths, '_cxcywh_n.txt')
    
    # Get images and label dirs
    train_images_dir = create_dir_if_not_exists(train_dir / 'images')
    train_labels_dir = create_dir_if_not_exists(train_dir / 'labels')
    val_images_dir = create_dir_if_not_exists(val_dir / 'images')
    val_labels_dir = create_dir_if_not_exists(val_dir / 'labels')
    
    # Build traindata
    for i, (td_in, td_bbox) in enumerate(zip(td_in_paths, td_bbox_paths)):
        shutil.copyfile(td_in, train_images_dir / f'{i}.png')
        with open(td_bbox, 'r') as file:
            td_bbox_lines = file.read().split('\n')
        td_bbox_lines.pop()
        with open(train_labels_dir / f'{i}.txt', "w") as text_file:
            for line in td_bbox_lines:
                text_file.write(f"0 {line}\n")
    print(f'Built {i} traindata!')
    
    # Build valdata
    for i, (vd_in, vd_bbox) in enumerate(zip(vd_in_paths, vd_bbox_paths)):
        shutil.copyfile(vd_in, val_images_dir / f'{i}.png')
        with open(vd_bbox, 'r') as file:
            vd_bbox_lines = file.read().split('\n')
        vd_bbox_lines.pop()
        with open(val_labels_dir / f'{i}.txt', "w") as text_file:
            for line in vd_bbox_lines:
                text_file.write(f"0 {line}\n")
    print(f'Built {i} valdata!')
    
    # Build yolov5 yaml
    yaml_path = dataset_dir / f'{dataset_name}.yaml'
    with open(yaml_path, "w") as text_file:
        text_file.write(f"path: {dataset_dir}\n")
        text_file.write(f"train: ./{train_dir_name}/images\n")
        text_file.write(f"val: ./{val_dir_name}/images\n")
        text_file.write(f"test:\n")
        text_file.write(f"\n")
        text_file.write(f"names:\n")
        text_file.write(f"    0: marker\n")

def get_adjacet_files_with_ending(file_paths: list[Path], ending):
    paths = []
    for fpath in file_paths:
        if type(fpath) is str:
            fpath = Path(fpath)
        paths.append(fpath.with_name(f'{"_".join(fpath.stem.split("_")[:-1])}{ending}'))
    return paths   

def get_files_from_folders_with_ending(folders, ending):
    paths = []
    for folder in folders:
        paths.extend(sorted(
            [
                os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(ending)
            ]
        ))
    return paths

def create_dir_if_not_exists(dir: Path, clear = False):
    if clear and os.path.isdir(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

if __name__ == '__main__':
    main()