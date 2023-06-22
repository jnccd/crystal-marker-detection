import argparse
import os
import math
import random
from pathlib import Path
import shutil
from timeit import default_timer as timer
from datetime import timedelta
import cv2

import numpy as np
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img, array_to_img

from cmd_tf.runconfigs import load_runconfig
from cmd_tf.utility import get_files_from_folders_with_ending

num_classes = 1
bbox_inflation = 5

def analyze(
    run_or_valdata: str,
    ):
    
    root_dir = Path(__file__).resolve().parent
    if os.path.exists(run_or_valdata) and os.path.isdir(run_or_valdata):
        valdata_path = Path(run_or_valdata)
    else:
        runs_dir = root_dir / 'runs'
        run_dir = runs_dir / f'{run_or_valdata}'
        valdata_path = run_dir / 'validation'
    
    ins = get_files_from_folders_with_ending([valdata_path], '_input.png')
    bbox_target_outs = get_files_from_folders_with_ending([valdata_path], '_target_output.txt')
    pic_target_outs = get_files_from_folders_with_ending([valdata_path], '_target_output.png')
    bbox_network_outs = get_files_from_folders_with_ending([valdata_path], '_network_output.txt')
    pic_network_outs = get_files_from_folders_with_ending([valdata_path], '_network_output.png')
    
    if len(bbox_network_outs) == 0:
        # Convert pic outs to bbox outs
        
        target_bboxes = []
        for target_out in pic_target_outs:
            img = cv2.imread(str(target_out), cv2.IMREAD_GRAYSCALE)
            target_bboxes.append(cluster_boxes_from_grayscale_img(img))
            
        network_bboxes = []
        for network_out in pic_network_outs:
            img = cv2.imread(str(network_out), cv2.IMREAD_GRAYSCALE)
            network_bboxes.append(cluster_boxes_from_grayscale_img(img))
    else:
        # Read bbox outs
        
        target_bboxes = []
        for target_out in bbox_target_outs:
            with open(target_out, 'r') as file:
                target_out_bbox_lines = file.readlines()
            target_out_bbox_lines.pop()
            bboxes_xyxy = []
            for line in target_out_bbox_lines:
                bboxes_xyxy.append(tuple([float(x) for x in line.split(' ')]))
            target_bboxes.append(bboxes_xyxy)
            
        network_bboxes = []
        for network_out in bbox_network_outs:
            with open(network_out, 'r') as file:
                network_out_bbox_lines = file.readlines()
            network_out_bbox_lines.pop()
            bboxes_xyxy = []
            for line in network_out_bbox_lines:
                bboxes_xyxy.append(tuple([float(x) for x in line.split(' ')]))
            network_bboxes.append(bboxes_xyxy)
            
    print('target_bboxes',target_bboxes)
    print('network_bboxes',network_bboxes)
            
    # TODO: Match target and pred bboxes
    
    # TODO: Measure performance using some metrics
    
    # TODO: Write out said metrics
    
def cluster_boxes_from_grayscale_img(img):
    global bbox_inflation
    
    bboxes_xyxy = []
    img_h, img_w = img.shape[:2]
    #sanity_check_img = np.zeros((img_h, img_w) + (3,), dtype = np.uint8)
    assert img is not None, "file could not be read!"
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,_ = cv2.findContours(thresh, 1, 2)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        x -= bbox_inflation
        y -= bbox_inflation
        w += bbox_inflation*2
        h += bbox_inflation*2
        bboxes_xyxy.append((x,y,x+w,y+h))
        #cv2.rectangle(sanity_check_img,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.imwrite(str(target_out)+'.cluster_test.png', sanity_check_img)
    
    return bboxes_xyxy