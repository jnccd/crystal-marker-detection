import argparse
import json
import os
import math
import random
from pathlib import Path
import shutil
import sys
import cv2
import re
from matplotlib import colors, pyplot as plt
from itertools import groupby
import numpy as np

from ensemble_boxes import *

from utility import *

parser = argparse.ArgumentParser(prog='fusion-inference', description='Performs fusion inference on multiple training run folders.')
parser.add_argument('-r','--runs-folder', type=str, help='Folders which contain the dataset folders in their subfolders which should be used for the inference group.')
parser.add_argument('-rnp','--run-name-pattern', type=str, help='Regex filter for run names that are part of the inference group.')
args = parser.parse_args()

run_name_pattern = re.compile(args.run_name_pattern)
runs_paths = get_all_subfolder_run_dirs([args.runs_folder])

# Get all paths of the runs that are chosen to be in the inference group
inference_group_runs_paths = [x for x in runs_paths if run_name_pattern.match(x['run_root'].stem)]

# For all results of every inference group run, build a data structure that contains relevant info
bboxes_per_img_per_inference_group = []
img_sizes = []
for inference_group_run_paths in inference_group_runs_paths:
    testdata_path = inference_group_run_paths['run_root'] / 'test'
    
    in_paths = get_files_from_folders_with_ending([testdata_path], '_input.png')
    bbox_target_out_paths = get_files_from_folders_with_ending([testdata_path], '_target_output.txt')
    bbox_network_out_paths = get_files_from_folders_with_ending([testdata_path], '_network_output.txt')
    
    bboxes_per_img = []
    for network_out_path, in_img_path in zip(bbox_network_out_paths, in_paths):
        in_img = cv2.imread(str(in_img_path))
        in_img_h, in_img_w = in_img.shape[:2]
        
        img_index = int(Path(network_out_path).stem.split('_')[0])
        # Build img sizes array
        while len(img_sizes) <= img_index:
            img_sizes.append(0)
        img_sizes[img_index] = (in_img_w, in_img_h)
        # Build pred arrays
        bboxes = []
        confs = []
        for line in read_textfile(network_out_path).split('\n'):
            if line.__contains__(' '):
                minx, miny, maxx, maxy, conf = [float(x) for x in line.split(' ')]
                bboxes.append([minx / in_img_w, 
                               miny / in_img_h, 
                               maxx / in_img_w, 
                               maxy / in_img_h])
                confs.append(conf)
        bboxes_per_img.append((img_index, bboxes, confs))
    bboxes_per_img.sort(key=lambda x: x[0])
    
    bboxes_per_img_per_inference_group.append(bboxes_per_img)
    
# Build the required data structure for ensemble_boxes for every image and write fuse result to network_output.txt
fused_test_out_path = create_dir_if_not_exists(inference_group_runs_paths[0]['run_root'] / 'test_fused', clear=True)
img_len = max([len(x) for x in bboxes_per_img_per_inference_group])
for i in range(img_len):
    bboxes_of_img_i_per_inference_group = [x[i] for x in bboxes_per_img_per_inference_group]
    
    boxes_list = [x[1] for x in bboxes_of_img_i_per_inference_group]
    scores_list = [x[2] for x in bboxes_of_img_i_per_inference_group]
    labels_list = [[0 for y in x[2]] for x in bboxes_of_img_i_per_inference_group]
    #print(boxes_list, '\n', scores_list, '\n', labels_list)
    
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list)
    #print(boxes, scores, labels)
    
    #print(list(zip(boxes, scores)))
    write_textfile('\n'.join(  [' '.join([str(float(x) * img_sizes[i][j % 2]) for j, x in enumerate(pred[0])]) + ' ' + str(pred[1]) 
                                for pred in zip(boxes, scores)]), 
                    fused_test_out_path / f'{i}_network_output.txt')

# Copy other files to test_fused
first_inf_group_test_folder_path = inference_group_runs_paths[0]['run_root'] / 'test'
in_paths = get_files_from_folders_with_ending([first_inf_group_test_folder_path], '_input.png')
bbox_target_out_paths = get_files_from_folders_with_ending([first_inf_group_test_folder_path], '_target_output.txt')
for i in range(img_len):
    shutil.copyfile(in_paths[i], fused_test_out_path / f'{i}_input.png')
    shutil.copyfile(bbox_target_out_paths[i], fused_test_out_path / f'{i}_target_output.txt')
