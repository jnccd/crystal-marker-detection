import argparse
import ast
import os
import random
import shutil
import sys
import time
import cv2
from pathlib import Path
from numpy import ndarray, uint8
from shapely import LineString, Point, Polygon

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
    parser.add_argument('-s','--size', type=int, help='Defines the image size for the dataset.')
    parser.add_argument('-tf','--traindata-folders', action='append', nargs='+', type=str, help='The folders containing train data.')
    parser.add_argument('-vf','--valdata-folders', action='append', nargs='+', type=str, help='The folders containing validation data.')
    parser.add_argument('-r','--ratio', type=float, help='Ratio of traindata to be assigned to valdata, if set overrides the -vf setting.')
    args = parser.parse_args()
    
    dataset_name = f'{args.type}-{args.name}'
    print(f'Creating {dataset_name}...')
    
    # --- Get Paths ---
    root_dir = Path(__file__).resolve().parent
    dataset_dir = create_dir_if_not_exists(root_dir / 'dataset' / dataset_name, clear=True)
    
    # Get td/vd folders
    td_folders = flatten(args.traindata_folders)
    if args.ratio is None:
        if args.valdata_folders is None:
            print('Error: I need valdata!')
            sys.exit(1)
        
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
    
    # Get target corners paths
    td_poly_paths = get_adjacent_files_with_ending(td_in_paths, '_vertices.txt')
    vd_poly_paths = get_adjacent_files_with_ending(vd_in_paths, '_vertices.txt')
    
    # --- Load dataseries ---
    td_in_imgs = [cv2.imread(str(p)) for p in td_in_paths]
    vd_in_imgs = [cv2.imread(str(p)) for p in vd_in_paths]
    td_target_polys = [[Polygon(b) for b in ast.literal_eval(read_textfile(p))] for p in td_poly_paths]
    vd_target_polys = [[Polygon(b) for b in ast.literal_eval(read_textfile(p))] for p in vd_poly_paths]
    
    # Resize and pad imgs and labels
    for i, (td_in_img, td_target_poly) in enumerate(zip(td_in_imgs, td_target_polys)):
        img, poly = resize_and_pad_with_labels(td_in_img, args.size, td_target_poly)
        td_in_imgs[i] = img
        td_target_polys[i] = poly
    for i, (vd_in_img, vd_target_poly) in enumerate(zip(vd_in_imgs, vd_target_polys)):
        img, poly = resize_and_pad_with_labels(vd_in_img, args.size, vd_target_poly)
        vd_in_imgs[i] = img
        vd_target_polys[i] = poly
    
    # --- Augment dataseries ---
    # TODO
    
    # --- Build dataset ---
    if args.type == 'seg':
        build_seg_dataset(td_in_imgs, td_target_polys, vd_in_imgs, vd_target_polys)
    elif args.type == 'yolov5':
        build_yolov5_dataset(td_in_imgs, td_target_polys, vd_in_imgs, vd_target_polys)
    elif args.type == 'csv':
        build_od_csv_dataset(td_in_imgs, td_target_polys, vd_in_imgs, vd_target_polys)
    else:
        print('Error: Unsupported dataset type!')
   
def build_seg_dataset(td_in_imgs: Mat, td_target_polys: Polygon, vd_in_imgs: Mat, vd_target_polys: Polygon):
    global train_dir_name, val_dir_name, dataset_name, dataset_dir
    
    train_dir = create_dir_if_not_exists(dataset_dir / train_dir_name)
    val_dir = create_dir_if_not_exists(dataset_dir / val_dir_name)
    
    for i, (td_in, td_polys) in enumerate(zip(td_in_imgs, td_target_polys)):
        cv2.imwrite(str(train_dir / f'{i}_in.png'), td_in)
        
        seg_image = np.zeros(td_in.shape[:2] + (3,), dtype = np.uint8)
        seg_image = rasterize_polys(seg_image, td_polys)
        
        cv2.imwrite(str(train_dir / f'{i}_seg.png'), seg_image)
    print(f'Built {i+1} traindata!')
    
    for i, (vd_in, vd_polys) in enumerate(zip(vd_in_imgs, vd_target_polys)):
        cv2.imwrite(str(val_dir / f'{i}_in.png'), vd_in)
        
        seg_image = np.zeros(vd_in.shape[:2] + (3,), dtype = np.uint8)
        seg_image = rasterize_polys(seg_image, vd_polys)
        
        cv2.imwrite(str(val_dir / f'{i}_seg.png'), seg_image)
    print(f'Built {i+1} valdata!')
    
def build_od_csv_dataset(td_in_imgs: Mat, td_target_polys: Polygon, vd_in_imgs: Mat, vd_target_polys: Polygon):
    global train_dir_name, val_dir_name, dataset_name, dataset_dir
    
    # Create train / val dirs
    train_dir = create_dir_if_not_exists(dataset_dir / train_dir_name)
    val_dir = create_dir_if_not_exists(dataset_dir / val_dir_name)
    
    # Get BBox files
    td_bbox_paths = get_adjacent_files_with_ending(td_in_paths, '_xyxy.txt')
    vd_bbox_paths = get_adjacent_files_with_ending(vd_in_paths, '_xyxy.txt')
    
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
    
def build_yolov5_dataset(td_in_imgs: Mat, td_target_polys: Polygon, vd_in_imgs: Mat, vd_target_polys: Polygon):
    global train_dir_name, val_dir_name, dataset_name, dataset_dir
    
    # Create train / val dirs
    train_dir = create_dir_if_not_exists(dataset_dir / train_dir_name)
    val_dir = create_dir_if_not_exists(dataset_dir / val_dir_name)
    
    # Get BBox files
    td_bbox_paths = get_adjacent_files_with_ending(td_in_paths, '_cxcywh_n.txt')
    vd_bbox_paths = get_adjacent_files_with_ending(vd_in_paths, '_cxcywh_n.txt')
    
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

if __name__ == '__main__':
    main()