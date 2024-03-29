import argparse
import ast
import json
import os
import random
import shutil
import sys
import time
import copy
import cv2
from pathlib import Path
from numpy import ndarray, uint8
from shapely import LineString, Point, Polygon, bounds
import albumentations as A

from utils import *

# low prio TODO: add test data support?
data_groups = ['train', 'val']
dataset_name = ''
dataset_dir = None
background_color = (114, 114, 114)
border_type = cv2.BORDER_CONSTANT

def main():
    global data_groups, dataset_name, dataset_dir, background_color, border_type
    
    parser = argparse.ArgumentParser(prog='dataset-creator', description='Combines multiple dataseries into a dataset.')
    parser.add_argument('-n','--name', type=str, help='Defines the (folder)name of the dataset.')
    parser.add_argument('-t','--type', type=str, help='Defines the type of dataset to be build, "seg" for segmentation, "yolov5" for yolov5 od (object detection), "csv" for csv od.')
    parser.add_argument('-s','--size', type=int, help='Defines the image size for the dataset, 0 for no resize.')
    parser.add_argument('-tf','--traindata-folders', action='append', nargs='+', type=str, help='The folders containing train data.')
    parser.add_argument('-vf','--valdata-folders', action='append', nargs='+', type=str, help='The folders containing validation data.')
    parser.add_argument('-taf','--target-folder', type=str, help='The folder to build the dataset folder into.')
    parser.add_argument('-r','--ratio', type=float, help='Ratio of traindata to be assigned to valdata, if set overrides the -vf setting.')
    parser.add_argument('-sd','--seed', type=int, help='Sets the seed that defines the pseudo random dataset generation.')
    parser.add_argument('-lv','--label-viz', action='store_true', help='If set wil draw the labels on the images, not for training, just for validation and for thesis examples.')
    
    parser.add_argument('-a','--augment', action='store_true', help='Augment the training data is some way.')
    parser.add_argument('-aim','--augment-img-multiplier', type=int, default=2, help='When augmenting multiply all images since they are augmented randomly to create more variation.') # TODO: Update desc
    
    parser.add_argument('-asgsc','--augment-smart-grid-shuffle-chance', type=float, default=0, help='Chance that smart-grid-shuffle is applied to a sample.')
    parser.add_argument('-apldc','--augment-label-dropout-chance', type=float, default=0, help='Chance that label-dropout is applied to a sample.')
    parser.add_argument('-apc','--augment-perspective-chance', type=float, default=0, help='Chance that perspective is applied to a sample.')
    parser.add_argument('-aps','--augment-perspective-strength', type=float, default=0, help='Augment perspective strength in percent, 1 results in the image becoming triangular.')
    parser.add_argument('-arc','--augment-rotation-chance', type=float, default=0, help='Chance that rotation is applied to a sample.')
    parser.add_argument('-ars','--augment-rotation-strength', type=float, default= 0, help='Maximum augment rotation in degrees.')
    parser.add_argument('-andrc','--augment-ninety-deg-rotation-chance', type=float, default=0, help='Chance that a 90, 180 or 270 deg rotation is applied to a sample.')
    parser.add_argument('-arcc','--augment-random-crop-chance', type=float, default=0, help='Chance that image is cropped randomly.')
    parser.add_argument('-arc2c','--augment-random-crop-v2-chance', type=float, default=0, help='Chance that image is cropped randomly. (Improved version)')
    parser.add_argument('-almc','--augment-label-move-chance', type=float, default=0, help='Chance that a label is moved randomly to another part of the image.')
    parser.add_argument('-alm2c','--augment-label-move-v2-chance', type=float, default=0, help='Chance that a label is moved randomly to another part of the image. (Improved version)')
    parser.add_argument('-abdc','--augment-black-dot-chance', type=float, default=0, help='Chance that black dots are placed on the image.')
    parser.add_argument('-alcc','--augment-label-curving-chance', type=float, default=0, help='Chance that labels are curved using a half circle function based vector field on the image.')
    
    parser.add_argument('-agnc','--augment-gauss-noise-chance', type=float, default=0, help='Chance that gauss noise is applied.')
    parser.add_argument('-agns','--augment-gauss-noise-strength', type=float, default=15, help='Strength of gauss noise that is applied.')
    args = parser.parse_args()
    
    if args.size is None:
        print('Please specify a size')
        sys.exit(1)
        
    # Set random seed
    rng_seed = args.seed if args.seed is not None else random.randrange(sys.maxsize)
    random.seed(rng_seed)
    
    args.name = args.name.replace('.','').replace('/','').replace('\\','').replace('>','').replace('<','').replace(':','').replace('|','').replace('?','').replace('*','')
    dataset_name = f'{args.type}-{args.size}-{args.name}'
    print(f'Creating {dataset_name}...')
    
    # --- Get Paths ---
    root_dir = Path(__file__).resolve().parent
    datasets_target_folder = root_dir / 'dataset' if not args.target_folder or args.target_folder.isspace() else Path(args.target_folder)
    dataset_dir = create_dir_if_not_exists(datasets_target_folder / dataset_name, clear=True)
    
    # Get td/vd folders
    td_folders = flatten(args.traindata_folders) if not args.traindata_folders is None else []
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
    
    # --- Augment dataseries ---
    aug_group = data_groups[0] # Augment only traindata
    if args.augment:
        aug_in_imgs = []
        aug_target_polys = []
        
        # Init Albumentations pixel level transforms, make sure to use own random here to use rng_seed
        transform = A.Compose([
            A.GaussNoise(var_limit=(0, args.augment_gauss_noise_strength), p=(1 if random.random() < args.augment_gauss_noise_chance else 0))
        ])
        
        for i, (in_img, target_poly) in enumerate(zip(in_imgs[aug_group], target_polys[aug_group])):
            for m in range(args.augment_img_multiplier):
                img_size_wh = tuple(reversed(in_img.shape[:2]))
                
                if m > 0:
                    aug_img = in_img.copy()
                    aug_polys = copy.deepcopy(target_poly)
                    
                    # Random Crop
                    if random.random() < args.augment_random_crop_chance:
                        aug_img, aug_polys = random_crop(aug_img, aug_polys, (args.size, args.size) if args.size != 0 else img_size_wh)
                        img_size_wh = tuple(reversed(aug_img.shape[:2]))
                        
                    # Random Crop V2
                    if random.random() < args.augment_random_crop_v2_chance:
                        aug_img, aug_polys = random_crop_v2(aug_img, aug_polys, (args.size, args.size) if args.size != 0 else img_size_wh)
                        img_size_wh = tuple(reversed(aug_img.shape[:2]))
                    
                    # Smart Grid Shuffle
                    if random.random() < args.augment_smart_grid_shuffle_chance:
                        aug_img, aug_polys = smart_grid_shuffle(aug_img, aug_polys, img_size_wh)
                    
                    # Poly Label Dropout
                    if random.random() < args.augment_label_dropout_chance:
                        aug_img, aug_polys = poly_label_dropout(aug_img, aug_polys)
                    
                    # Poly Label Move
                    if random.random() < args.augment_label_move_chance:
                        aug_img, aug_polys = poly_label_move(aug_img, aug_polys)
                        
                    # Poly Label Move v2
                    if random.random() < args.augment_label_move_v2_chance:
                        aug_img, aug_polys = poly_label_move_v2(aug_img, aug_polys)
                        
                    # Black dot aug
                    if random.random() < args.augment_black_dot_chance:
                        aug_img, aug_polys = black_dot_aug(aug_img, aug_polys)
                        
                    # Poly label curving
                    if random.random() < args.augment_label_curving_chance:
                        aug_img, aug_polys = poly_label_curving(aug_img, aug_polys)
                    
                    # Matrix Transform
                    mats = []
                    # -- Perspective
                    if random.random() < args.augment_perspective_chance:
                        mats.append(create_random_persp_mat(img_size_wh, perspective_strength=args.augment_perspective_strength))
                    # -- Rotation
                    rotation_angle = 0
                    if random.random() < args.augment_rotation_chance and args.augment_rotation_strength > 0:
                        rotation_angle += (random.random() - 0.5) * 2 * args.augment_rotation_strength
                    if random.random() < args.augment_ninety_deg_rotation_chance:
                        rotation_angle += random.randrange(0, 4) * 90
                    mats.append(
                        # Make affine rot mat 3x3
                        np.vstack([
                            cv2.getRotationMatrix2D(
                                (img_size_wh[0]/2, img_size_wh[1]/2), 
                                rotation_angle, 
                                1), 
                            np.array([0, 0, 1])
                        ])
                    )
                    # -- Apply
                    final_mat = np.identity(3)
                    mats.reverse()
                    for mat in mats:
                        final_mat = final_mat @ mat
                    aug_img, aug_polys = homogeneous_mat_transform(aug_img, aug_polys, img_size_wh, final_mat, background_color=background_color, border_type=border_type)
                    
                    # Apply Albumentations transforms
                    aug_img = transform(image=aug_img)["image"]
                    
                    aug_in_imgs.append(aug_img)
                    aug_target_polys.append(aug_polys)
                else:
                    # Keep image original at least once
                    aug_in_imgs.append(in_img)
                    aug_target_polys.append(target_poly)
                
        in_imgs[aug_group] = aug_in_imgs
        target_polys[aug_group] = aug_target_polys
        
    # Resize and pad imgs and labels
    if args.size > 0:
        for group in data_groups:
            for i, (in_img, target_poly) in enumerate(zip(in_imgs[group], target_polys[group])):
                img, poly = resize_and_pad_with_labels(in_img, args.size, target_poly, background_color, border_type)
                in_imgs[group][i] = img
                target_polys[group][i] = poly
                
    # Build debug out labels on imgs
    if args.label_viz > 0:
        for group in data_groups:
            for i, (in_img, target_poly) in enumerate(zip(in_imgs[group], target_polys[group])):
                for poly in target_poly:
                    pts = np.array([(int(point[0]), int(point[1])) for point in poly.exterior.coords[:-1]], dtype=np.int32)
                    in_img = cv2.polylines(in_img, pts=[pts], isClosed=True, color=(0,0,255))
                in_imgs[group][i] = in_img
    
    # --- Build dataset ---
    if args.type == 'seg':
        build_seg_dataset(in_imgs, target_polys)
    elif args.type == 'yolov5':
        build_yolov5_dataset(in_imgs, target_polys)
    elif args.type == 'coco':
        build_coco_dataset(in_imgs, target_polys)
    elif args.type == 'csv':
        build_od_csv_dataset(in_imgs, target_polys)
    elif args.type == 'pet':
        build_pet_dataset(in_imgs, target_polys)
    elif args.type == 'segpet':
        build_segpet_dataset(in_imgs, target_polys)
    elif args.type == 'alb-test':
        build_alb_test_dataset(in_imgs, target_polys)
    else:
        print('Error: Unsupported dataset type!') 
        exit()
        
    # Add dataset definition dict
    write_textfile(json.dumps({
            'name': dataset_name,
            'type': args.type,
            'rng_seed': rng_seed,
            'td_series': args.traindata_folders,
            'vd_series': args.valdata_folders,
            'ratio': args.ratio,
            'command': 'python ' + ' '.join(sys.argv),
            'augment': args.augment,
            'augment_img_mult': args.augment_img_multiplier,
            'smart_grid_shuffle_chance': args.augment_smart_grid_shuffle_chance,
            'label_dropout_chance': args.augment_label_dropout_chance,
            'perspective_chance': args.augment_perspective_chance,
            'perspective_strength': args.augment_perspective_strength,
            'rotation_chance': args.augment_rotation_chance,
            'rotation_strength': args.augment_rotation_strength,
            'ninety_deg_rotation_chance': args.augment_ninety_deg_rotation_chance,
            'random_crop_chance': args.augment_random_crop_chance,
            'random_crop_v2_chance': args.augment_random_crop_v2_chance,
            'label_move_chance': args.augment_label_move_chance,
            'label_move_v2_chance': args.augment_label_move_v2_chance,
            'gauss_noise_chance': args.augment_gauss_noise_chance,
            'black_dot_chance': args.augment_black_dot_chance,
            'label_curving_chance': args.augment_label_curving_chance
        }, indent=4), dataset_dir / 'dataset-def.json')

def build_seg_dataset(in_imgs, target_polys):
    global data_groups, dataset_name, dataset_dir
    
    # Create train / val dirs
    dir = {}
    for group in data_groups:
        dir[group] = create_dir_if_not_exists(dataset_dir / group)
    
    # Build groups data
    i = -1
    for group in data_groups:
        for i, (in_img, polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            cv2.imwrite(str(dir[group] / f'{i}_in.png'), in_img)
            
            seg_image = np.zeros(in_img.shape[:2] + (3,), dtype = np.uint8)
            seg_image = rasterize_polys(seg_image, polys)
            
            cv2.imwrite(str(dir[group] / f'{i}_seg.png'), seg_image)
        print(f'Built {i+1} {group}data!')
        
def build_alb_test_dataset(in_imgs, target_polys):
    global data_groups, dataset_name, dataset_dir
    
    # Create train / val dirs
    dir = {}
    for group in data_groups:
        dir[group] = create_dir_if_not_exists(dataset_dir / group)
        
    transform = A.Compose([
        A.ElasticTransform(always_apply=True)
    ], bbox_params=A.BboxParams(format='pascal_voc'))
    
    # Build groups data
    i = -1
    for group in data_groups:
        for i, (in_img, polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            h, w = in_img.shape[:2]
            polys: List[Polygon] = polys
            
            #bboxes = [[c / (w if i % 2 == 0 else h) for i, c in enumerate(p.bounds)] for p in polys]
            bboxes = [p.bounds for p in polys]
            bboxes = [[int(x) for x in b] for b in bboxes]
            bboxes = [(b[0], b[1], b[2], b[3], 'marker') for b in bboxes]
            transform_out = transform(image=in_img, bboxes=bboxes)
            aug_img = transform_out["image"]
            aug_bboxes = transform_out["bboxes"]
            
            cv2.imwrite(str(dir[group] / f'{i}_in.png'), aug_img)
            
            seg_image = np.zeros(in_img.shape[:2] + (3,), dtype = np.uint8)
            for bbox in aug_bboxes:
                bbox = bbox[:4]
                bbox = [int(x) for x in bbox]
                cv2.rectangle(seg_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            cv2.imwrite(str(dir[group] / f'{i}_seg.png'), seg_image)
        print(f'Built {i+1} {group}data!')
        
def build_pet_dataset(in_imgs, target_polys):
    global data_groups, dataset_name, dataset_dir, background_color, border_type
    pet_target_size = 160
    
    # Create train / val dirs
    dir = {}
    for group in data_groups:
        dir[group] = create_dir_if_not_exists(dataset_dir / group)
    
    # Build mixed groups data
    mixed_group_data = [] # [img_path, img, vertices, verts_path]
    for group in data_groups:
        for i, (in_img, polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            for j, poly in enumerate(polys):
                # Rotate
                in_img_size_wh = tuple(reversed(in_img.shape[:2]))
                rotation_angle = (random.random() - 0.5) * 2 * 270
                rot_in_img, [poly] = homogeneous_mat_transform(in_img, [poly], in_img_size_wh, 
                    cv2.getRotationMatrix2D(
                        (poly.centroid.x, poly.centroid.y), 
                        rotation_angle, 
                        1), 
                    background_color=background_color, 
                    border_type=border_type, 
                    min_label_visiblity=0.01)
                
                # Prepare bbox
                poly: Polygon = poly # For linting
                b = [int(x) for x in inflate_bbox_xyxy(poly.bounds, 0.3)]
                bw = b[3] - b[1]
                bh = b[2] - b[0]
                bdiff = abs(bw - bh)
                if bw > bh:
                    b[0] -= int(bdiff/2)
                    b[2] += int(bdiff/2)
                else:
                    b[1] -= int(bdiff/2)
                    b[3] += int(bdiff/2)
                
                # Prepare img cutout
                crop_img = rot_in_img[b[1]:b[3], b[0]:b[2]]
                if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                    continue
                poly: Polygon = transform(poly, lambda x: np.array([(p[0] - b[0], p[1] - b[1]) for p in x] ))
                # Resize and pad
                crop_img, [poly] = resize_and_pad_with_labels(crop_img, pet_target_size, [poly], background_color, border_type)
                
                # Convert poly to seg img
                seg_image = np.zeros(crop_img.shape[:2] + (1,), dtype = np.uint8)
                seg_image = rasterize_polys(seg_image, [poly])
                
                # Store cutout output in mixed_group_data
                mixed_group_data.append(
                    (
                        f'{i}_{j}_in.png', 
                        crop_img, 
                        str(poly.exterior.coords[:-1]), 
                        f'{i}_{j}_p.txt',
                        seg_image, 
                        f'{i}_{j}_seg.png'
                    )
                )
        
    # Shuffle mixed group data, regroup, write
    random.Random(42).shuffle(mixed_group_data)
    i = 0
    for group in data_groups:
        shuffled_group_data = mixed_group_data[i:i+sum([len(x) for x in target_polys[group]])]
        
        for data in shuffled_group_data:
            cv2.imwrite(str(dir[group] / data[0]), data[1])
            write_textfile(data[2], dir[group] / data[3])
            #cv2.imwrite(str(dir[group] / data[5]), data[4])
        
        i += len(in_imgs[group])
        
        print(f'Built {len(in_imgs[group])} {group}data!')
        
def build_segpet_dataset(in_imgs, target_polys):
    global data_groups, dataset_name, dataset_dir, background_color, border_type
    pet_target_size = 160
    
    # Create train / val dirs
    dir = {}
    for group in data_groups:
        dir[group] = create_dir_if_not_exists(dataset_dir / group)
    
    # Build mixed groups data
    mixed_group_data = [] # [img_path, img, vertices, verts_path]
    for group in data_groups:
        for i, (in_img, polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            for j, poly in enumerate(polys):
                # Rotate
                in_img_size_wh = tuple(reversed(in_img.shape[:2]))
                rotation_angle = (random.random() - 0.5) * 2 * 270
                rot_in_img, [poly] = homogeneous_mat_transform(in_img, [poly], in_img_size_wh, 
                    cv2.getRotationMatrix2D(
                        (poly.centroid.x, poly.centroid.y), 
                        rotation_angle, 
                        1), 
                    background_color=background_color, 
                    border_type=border_type, 
                    min_label_visiblity=0.01)
                
                # Prepare bbox
                poly: Polygon = poly # For linting
                b = [int(x) for x in inflate_bbox_xyxy(poly.bounds, 0.3)]
                bw = b[3] - b[1]
                bh = b[2] - b[0]
                bdiff = abs(bw - bh)
                if bw > bh:
                    b[0] -= int(bdiff/2)
                    b[2] += int(bdiff/2)
                else:
                    b[1] -= int(bdiff/2)
                    b[3] += int(bdiff/2)
                
                # Prepare img cutout
                crop_img = rot_in_img[b[1]:b[3], b[0]:b[2]]
                if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                    continue
                poly: Polygon = transform(poly, lambda x: np.array([(p[0] - b[0], p[1] - b[1]) for p in x] ))
                
                # Resize and pad
                crop_img, [poly] = resize_and_pad_with_labels(crop_img, pet_target_size, [poly], background_color, border_type)
                
                # Convert poly to seg img
                seg_image = np.zeros(crop_img.shape[:2] + (1,), dtype = np.uint8)
                seg_image = rasterize_polys(seg_image, [poly])
                # pts = np.array([(int(point[0]), int(point[1])) for point in poly.exterior.coords[:-1]], dtype=np.int32)
                # seg_image = cv2.polylines(seg_image, pts=[pts], isClosed=True, color=255)
                # seg_image = cv2.GaussianBlur(seg_image, (9,9), 0)
                # max_brightness = np.max(seg_image)
                # seg_image = (seg_image.astype('float32') * (255 / max_brightness)).astype('uint8')
                
                # Store cutout output in mixed_group_data
                mixed_group_data.append(
                    (
                        f'{i}_{j}_in.png', 
                        crop_img, 
                        seg_image, 
                        f'{i}_{j}_seg.png'
                    )
                )
        
    # Shuffle mixed group data, regroup, write
    random.Random(42).shuffle(mixed_group_data)
    i = 0
    for group in data_groups:
        shuffled_group_data = mixed_group_data[i:i+sum([len(x) for x in target_polys[group]])]
        
        for data in shuffled_group_data:
            cv2.imwrite(str(dir[group] / data[0]), data[1])
            cv2.imwrite(str(dir[group] / data[3]), data[2])
        
        i += len(in_imgs[group])
        
        print(f'Built {len(in_imgs[group])} {group}data!')
        
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
        
def build_coco_dataset(in_imgs, target_polys):
    global data_groups, dataset_name, dataset_dir
    
    annot_dir = create_dir_if_not_exists(dataset_dir / 'annotations')
    img_dir = create_dir_if_not_exists(dataset_dir / 'images')
    
    # Get images and annots paths
    annot_file = {}
    images_dir = {}
    for group in data_groups:
        annot_file[group] = annot_dir / f'instances_{group}2017.json'
        images_dir[group] = create_dir_if_not_exists(img_dir / f'{group}2017')
        
    # Build groups data
    for group in data_groups:
        group_annot_dict = {
            "info": {
                "description": dataset_name
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"supercategory": "marker","id": 1,"name": "marker"}],
            "segment_info": []
        }
        
        for i, (td_in, td_polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            img_h, img_w = td_in.shape[:2]
            pic_path: Path = images_dir[group] / f'{i}.png'
            cv2.imwrite(str(pic_path), td_in)
            
            group_annot_dict["images"].append({
                "id":i,
                "width":img_w,
                "height":img_h,
                "file_name":pic_path.stem + pic_path.suffix
            })
            
            img_bounds = (0,0,img_w,img_h)
            xyxy_bboxes = [bounds(poly) for poly in td_polys]
            xyxy_bboxes = [keep_box_in_bounds(bbox, img_bounds) for bbox in xyxy_bboxes]
            xywh_bboxes = [[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in xyxy_bboxes]
            for ib, (bbox, poly) in enumerate(zip(xywh_bboxes, td_polys)):
                poly: Polygon = poly # Linting
                group_annot_dict["annotations"].append({
                    "id": i*100000+ib,
                    "image_id": i,
                    "segmentation": [list(flatten(poly.exterior.coords[:-1]))],
                    "bbox": list(bbox),
                    "area": poly.area,
                    "category_id": 1,
                    "iscrowd": 0,
                })
            
        write_textfile(json.dumps(group_annot_dict), annot_file[group])
        print(f'Built {len(in_imgs[group])} {group}data!')
    
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
        i = 0
        for i, (td_in, td_polys) in enumerate(zip(in_imgs[group], target_polys[group])):
            img_h, img_w = td_in.shape[:2]
            pic_path = images_dir[group] / f'{i}.png'
            cv2.imwrite(str(pic_path), td_in)
            
            img_bounds = (3,3,img_w,img_h)
            xyxy_bboxes = [bounds(poly) for poly in td_polys]
            xyxy_bboxes = [keep_box_in_bounds(bbox, img_bounds) for bbox in xyxy_bboxes]
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