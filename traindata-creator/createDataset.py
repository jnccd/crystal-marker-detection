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
    parser.add_argument('-t','--type', type=str, help='Defines the type of dataset to be build, either "seg" for segmentation or "yolov5" for yolov5 object detection.')
    parser.add_argument('-tf','--traindata-folders', type=str, help='The folders containing train data, separated by a #.')
    parser.add_argument('-vf','--valdata-folders', type=str, help='The folders containing validation data, separated by a #.')
#    parser.add_argument('-r','--ratio', type=str, help='Ratio of traindata to be assigned to valdata.')
    args = parser.parse_args()
    
    train_dir_name = 'train'
    val_dir_name = 'val'
    
    dataset_name = f'dataset-{args.type}-{args.name}'
    print(f'Creating {dataset_name}...')
    
    # --- Get Folders ---
    td_folders = args.traindata_folders.split('#')
    vd_folders = args.valdata_folders.split('#')
    
    td_in_paths = get_files_from_folders_with_ending(td_folders, '_in.png')
    vd_in_paths = get_files_from_folders_with_ending(vd_folders, '_in.png')
    
    root_dir = Path(__file__).resolve().parent
    dataset_dir = create_dir_if_not_exists(root_dir / dataset_name, clear=True)
    
    train_dir = create_dir_if_not_exists(dataset_dir / train_dir_name)
    val_dir = create_dir_if_not_exists(dataset_dir / val_dir_name)
    
    # --- Build dataset ---
    # Simple segmentation dataset
    if args.type == 'seg': 
        td_seg_paths = get_files_from_folders_with_ending(td_folders, '_seg.png')
        vd_seg_paths = get_files_from_folders_with_ending(vd_folders, '_seg.png')
        
        for i, (td_in, td_seg) in enumerate(zip(td_in_paths, td_seg_paths)):
            shutil.copyfile(td_in, train_dir / f'{i}_in.png')
            shutil.copyfile(td_seg, train_dir / f'{i}_seg.png')
        print(f'Built {i} traindata!')
        
        for i, (vd_in, vd_seg) in enumerate(zip(vd_in_paths, vd_seg_paths)):
            shutil.copyfile(vd_in, val_dir / f'{i}_in.png')
            shutil.copyfile(vd_seg, val_dir / f'{i}_seg.png')
        print(f'Built {i} valdata!')
    # Yolov5 Style Dataset
    elif args.type == 'yolov5': 
        td_bbox_paths = get_files_from_folders_with_ending(td_folders, '_cxcywh_n.txt')
        vd_bbox_paths = get_files_from_folders_with_ending(vd_folders, '_cxcywh_n.txt')
        
        train_images_dir = create_dir_if_not_exists(train_dir / 'images')
        train_labels_dir = create_dir_if_not_exists(train_dir / 'labels')
        
        val_images_dir = create_dir_if_not_exists(val_dir / 'images')
        val_labels_dir = create_dir_if_not_exists(val_dir / 'labels')
        
        for i, (td_in, td_bbox) in enumerate(zip(td_in_paths, td_bbox_paths)):
            shutil.copyfile(td_in, train_images_dir / f'{i}.png')
            with open(td_bbox, 'r') as file:
                td_bbox_lines = file.read().split('\n')
            td_bbox_lines.pop()
            with open(train_labels_dir / f'{i}.txt', "w") as text_file:
                for line in td_bbox_lines:
                    text_file.write(f"0 {line}\n")
        print(f'Built {i} traindata!')
        
        for i, (vd_in, vd_bbox) in enumerate(zip(vd_in_paths, vd_bbox_paths)):
            shutil.copyfile(vd_in, val_images_dir / f'{i}.png')
            with open(vd_bbox, 'r') as file:
                vd_bbox_lines = file.read().split('\n')
            vd_bbox_lines.pop()
            with open(val_labels_dir / f'{i}.txt', "w") as text_file:
                for line in vd_bbox_lines:
                    text_file.write(f"0 {line}\n")
        print(f'Built {i} valdata!')
        
        yaml_path = dataset_dir / f'{dataset_name}.yaml'
        with open(yaml_path, "w") as text_file:
            text_file.write(f"path: .\n")
            text_file.write(f"train: ./{train_dir_name}/images\n")
            text_file.write(f"val: ./{val_dir_name}/images\n")
            text_file.write(f"test:\n")
            text_file.write(f"\n")
            text_file.write(f"names:\n")
            text_file.write(f"\t0: marker\n")
    
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