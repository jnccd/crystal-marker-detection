import argparse
import os
import math
import random
from pathlib import Path
import shutil
from timeit import default_timer as timer
from datetime import timedelta

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

def analyze(
    run_or_valdata: str = [],
    ):
    
    root_dir = Path(__file__).resolve().parent
    if os.path.exists(run_or_valdata) and os.path.isdir(run_or_valdata):
        valdata_path = Path(run_or_valdata)
    else:
        runs_dir = root_dir / 'runs'
        run_dir = runs_dir / f'run-{run_or_valdata}'
        valdata_path = run_dir / 'validation'
    
    ins = get_files_from_folders_with_ending(valdata_path, '_input.png')
    bbox_target_outs = get_files_from_folders_with_ending(valdata_path, '_target_output.txt')
    pic_target_outs = get_files_from_folders_with_ending(valdata_path, '_target_output.png')
    bbox_network_outs = get_files_from_folders_with_ending(valdata_path, '_network_output.txt')
    pic_network_outs = get_files_from_folders_with_ending(valdata_path, '_network_output.png')
    
    if len(bbox_network_outs) == 0:
        # TODO: Convert pic outs in bbox outs
        zip(pic_target_outs, pic_network_outs)
    else:
        for target_out, network_out in zip(bbox_target_outs, bbox_network_outs):
            with open(network_out, 'r') as file:
                network_out_bbox_lines = file.readlines()
            network_out_bbox_lines.pop()
            
            with open(target_out, 'r') as file:
                target_out_bbox_lines = file.readlines()
            target_out_bbox_lines.pop()
            
    # TODO: Match target and pred bboxes
    
    # TODO: Measure performance using some metrics
    
    # TODO: Write out said metrics
    