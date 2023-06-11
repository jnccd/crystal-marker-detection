import argparse
import os
import shutil
import sys
import time
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog='dataset-creator', description='Combines multiple dataseries to a dataset.')
    parser.add_argument('-n','--name', type=str, help='Defines the (folder)name of the dataset.')
    parser.add_argument('-t','--type', type=str, help='Defines the type of dataset to be build, either "seg" for segmentation or "od" for object detection.')
    parser.add_argument('-tf','--traindata-folders', type=str, help='The folders containing train data, separated by a #.')
    parser.add_argument('-vf','--valdata-folders', type=str, help='The folders containing validation data, separated by a #.')
#    parser.add_argument('-r','--ratio', type=str, help='Ratio of traindata to be assigned to valdata.')
    args = parser.parse_args()
    
    td_folders = args.traindata_folders.split('#')
    vd_folders = args.valdata_folders.split('#')
    
    print("Traindata folders: ", td_folders)
    print("Valdata folders: ", vd_folders)
    
    td_in_paths = get_files_from_folders_with_ending(td_folders, '_in.png')
    vd_in_paths = get_files_from_folders_with_ending(vd_folders, '_in.png')
    
    print("Found Traindata: ", td_in_paths)
    print("Found Valdata: ", vd_in_paths)
    
    root_dir = Path(__file__).resolve().parent
    dataset_dir = create_dir_if_not_exists(root_dir / ('dataset-' + args.name), clear=True)
    
    train_dir = create_dir_if_not_exists(dataset_dir / 'train')
    val_dir = create_dir_if_not_exists(dataset_dir / 'val')
    
    if args.type == 'seg':
        td_seg_paths = get_files_from_folders_with_ending(td_folders, '_seg.png')
        vd_seg_paths = get_files_from_folders_with_ending(vd_folders, '_seg.png')
        
        for i, (td_in, td_seg) in enumerate(zip(td_in_paths, td_seg_paths)):
            shutil.copyfile(td_in, train_dir / f'{i}_in.png')
            shutil.copyfile(td_seg, train_dir / f'{i}_seg.png')
            
        for i, (vd_in, vd_seg) in enumerate(zip(vd_in_paths, vd_seg_paths)):
            shutil.copyfile(vd_in, val_dir / f'{i}_in.png')
            shutil.copyfile(vd_seg, val_dir / f'{i}_seg.png')
    
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