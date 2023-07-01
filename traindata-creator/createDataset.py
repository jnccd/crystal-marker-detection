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
from shapely import LineString, Point, Polygon, bounds

from utils import *

# low prio TODO: add test data support?
data_groups = ['train', 'val']
dataset_name = ''
dataset_dir = None
background_color = (114, 114, 114)
border_type = cv2.BORDER_REPLICATE

def main():
    global data_groups, dataset_name, dataset_dir, background_color, border_type
    
    parser = argparse.ArgumentParser(prog='dataset-creator', description='Combines multiple dataseries into a dataset.')
    parser.add_argument('-n','--name', type=str, help='Defines the (folder)name of the dataset.')
    parser.add_argument('-t','--type', type=str, help='Defines the type of dataset to be build, "seg" for segmentation, "yolov5" for yolov5 od (object detection), "csv" for csv od.')
    parser.add_argument('-s','--size', type=int, help='Defines the image size for the dataset.')
    parser.add_argument('-tf','--traindata-folders', action='append', nargs='+', type=str, help='The folders containing train data.')
    parser.add_argument('-vf','--valdata-folders', action='append', nargs='+', type=str, help='The folders containing validation data.')
    parser.add_argument('-r','--ratio', type=float, help='Ratio of traindata to be assigned to valdata, if set overrides the -vf setting.')
    parser.add_argument('-a','--augment', action='store_true', help='Augment the training data is some way.')
    parser.add_argument('-aim','--augment-img-multiplier', type=int, default=1, help='When augmenting multiply all images since they are augmented randomly to create more variation.')
    args = parser.parse_args()
    
    if args.size is None:
        print('Please specify a size')
        sys.exit(1)
    
    dataset_name = f'{args.type}-{args.size}{"-aug" if args.augment else ""}-{args.name}'
    print(f'Creating {dataset_name}...')
    
    # --- Get Paths ---
    root_dir = Path(__file__).resolve().parent
    dataset_dir = create_dir_if_not_exists(root_dir / 'dataset' / dataset_name, clear=True)
    
    # Get td/vd folders
    td_folders = flatten(args.traindata_folders)
    in_paths = {}
    if args.ratio is None:
        if args.valdata_folders is None:
            print('Error: I need valdata!')
            sys.exit(1)
        
        vd_folders = flatten(args.valdata_folders)
        
        in_paths['train'] = get_files_from_folders_with_ending(td_folders, '_in.png')
        in_paths['val'] = get_files_from_folders_with_ending(vd_folders, '_in.png')
    else:
        full_td_in_paths = get_files_from_folders_with_ending(td_folders, '_in.png')
        random.Random(42).shuffle(full_td_in_paths)
        
        n = len(full_td_in_paths)
        nt = int(n * (1-args.ratio))
        
        in_paths['train'] = full_td_in_paths[:nt]
        in_paths['val'] = full_td_in_paths[nt:]
    
    # Get target corners paths
    poly_paths = {}
    for group in data_groups:
        poly_paths[group] = get_adjacent_files_with_ending(in_paths[group], '_vertices.txt')
    
    # --- Load dataseries ---
    in_imgs = {}
    target_polys = {}
    for group in data_groups:
        in_imgs[group] = [cv2.imread(str(p)) for p in in_paths[group]]
        target_polys[group] = [[Polygon(b) for b in ast.literal_eval(read_textfile(p))] for p in poly_paths[group]]
    
    # Resize and pad imgs and labels
    for group in data_groups:
        for i, (in_img, target_poly) in enumerate(zip(in_imgs[group], target_polys[group])):
            img, poly = resize_and_pad_with_labels(in_img, args.size, target_poly, background_color, border_type)
            in_imgs[group][i] = img
            target_polys[group][i] = poly
    
    # --- Augment dataseries ---
    aug_group = data_groups[0] # Augment only traindata
    if args.augment:
        aug_in_imgs = []
        aug_target_polys = []
        
        for i, (in_img, target_poly) in enumerate(zip(in_imgs[aug_group], target_polys[aug_group])):
            for m in range(args.augment_img_multiplier):
                img_size = (args.size, args.size)
                
                aug_img, aug_polys = smart_grid_shuffle(in_img, target_poly, img_size)
                
                # Matrix transform img
                persp_mat = create_random_persp_mat((args.size, args.size))
                rot_mat_3d = np.vstack([cv2.getRotationMatrix2D((img_size[0]/2, img_size[1]/2), random.randrange(0, 360), 1), np.array([0, 0, 1])])
                final_mat = persp_mat @ rot_mat_3d
                aug_img, aug_polys = homogeneous_mat_transform(aug_img, aug_polys, img_size, final_mat, border_type=border_type)
                
                aug_in_imgs.append(aug_img)
                aug_target_polys.append(aug_polys)
                
        in_imgs[aug_group] = aug_in_imgs
        target_polys[aug_group] = aug_target_polys
    
    # --- Build dataset ---
    if args.type == 'seg':
        build_seg_dataset(in_imgs, target_polys)
    elif args.type == 'yolov5':
        build_yolov5_dataset(in_imgs, target_polys)
    elif args.type == 'csv':
        build_od_csv_dataset(in_imgs, target_polys)
    else:
        print('Error: Unsupported dataset type!')
   
def build_seg_dataset(in_imgs, target_polys):
    global data_groups, dataset_name, dataset_dir
    
    # Create train / val dirs
    dir = {}
    for group in data_groups:
        dir[group] = create_dir_if_not_exists(dataset_dir / group)
    
    # Build groups data
    for group in data_groups:
        for i, (in_img, polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            cv2.imwrite(str(dir[group] / f'{i}_in.png'), in_img)
            
            seg_image = np.zeros(in_img.shape[:2] + (3,), dtype = np.uint8)
            seg_image = rasterize_polys(seg_image, polys)
            
            cv2.imwrite(str(dir[group] / f'{i}_seg.png'), seg_image)
        print(f'Built {i+1} {group}data!')
    
def build_od_csv_dataset(in_imgs, target_polys):
    global data_groups, dataset_name, dataset_dir
    
    # Create train / val dirs
    dir = {}
    for group in data_groups:
        dir[group] = create_dir_if_not_exists(dataset_dir / group)
    
    # Build groups data
    for group in data_groups:
        csv_entries = []
        csv_path = dataset_dir / f'{group}.csv'
        for i, (in_img, polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            pic_path = dir[group] / f'{i}.png'
            cv2.imwrite(str(pic_path), in_img)
            
            bboxes = [[int(x) for x in bounds(poly)] for poly in polys]
            for box in bboxes:
                csv_entries.append(f'{str(pic_path)},{",".join([str(x) for x in box])},marker') # Low Prio TODO: Add more classes
        with open(csv_path, "w") as text_file:
            for entry in csv_entries:
                text_file.write(f"{entry}\n")
        print(f'Built {i} {group}data!')
    
    classes_csv_path = dataset_dir / 'classes.csv'
    with open(classes_csv_path, "w") as text_file:
        text_file.write(f"marker,0\n")
    
def build_yolov5_dataset(in_imgs, target_polys):
    global data_groups, dataset_name, dataset_dir
    
    # Create group dirs
    dir = {}
    for group in data_groups:
        dir[group] = create_dir_if_not_exists(dataset_dir / group)
    
    # Get images and label dirs
    images_dir = {}
    labels_dir = {}
    for group in data_groups:
        images_dir[group] = create_dir_if_not_exists(dir[group] / 'images')
        labels_dir[group] = create_dir_if_not_exists(dir[group] / 'labels')
    
    # Build groups data
    for group in data_groups:
        for i, (td_in, td_polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            img_h, img_w = td_in.shape[:2]
            pic_path = images_dir[group] / f'{i}.png'
            cv2.imwrite(str(pic_path), td_in)
            
            xyxy_bboxes = [bounds(poly) for poly in td_polys]
            xywh_bboxes = [[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in xyxy_bboxes]
            xywh_n_bboxes = [[bbox[0] / img_w, bbox[1] / img_h, bbox[2] / img_w, bbox[3] / img_h] for bbox in xywh_bboxes]
            cxcywh_n_bboxes = [[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, bbox[2], bbox[3]] for bbox in xywh_n_bboxes]
            
            with open(labels_dir[group] / f'{i}.txt', "w") as text_file:
                for bbox in cxcywh_n_bboxes:
                    text_file.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        print(f'Built {i} {group}data!')
    
    # Build yolov5 yaml
    yaml_path = dataset_dir / f'{dataset_name}.yaml'
    with open(yaml_path, "w") as text_file:
        text_file.write(f"path: {dataset_dir}\n")
        text_file.write(f"train: ./{data_groups[0]}/images\n")
        text_file.write(f"val: ./{data_groups[1]}/images\n")
        text_file.write(f"test:\n")
        text_file.write(f"\n")
        text_file.write(f"names:\n")
        text_file.write(f"    0: marker\n")

if __name__ == '__main__':
    main()