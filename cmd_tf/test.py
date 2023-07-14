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
from cmd_tf.utility import *

num_classes = 1

def test(
    testdata,
    run: str = 'default', 
    size: int = 320, 
    extra_settings = {},
    ):
    
    print(f'Testing {run}...')
    img_size = (size, size)
    
    root_dir = Path(__file__).resolve().parent
    runs_dir = root_dir / 'runs'
    run_dir = runs_dir / f'{run}'
    test_dir = create_dir_if_not_exists(run_dir / 'test', clear=True)
    weights_dir = run_dir / 'weights'
    
    testdata_path = Path(testdata)
    testdata_paths = []
    targetdata_paths = []
    if testdata_path.is_dir():
        if len(get_files_from_folders_with_ending([testdata_path], (".json"))) > 0:
            print(f'Got dataset input')
            val_folder_path = testdata_path / 'val'
            testdata_paths = get_files_from_folders_with_ending([val_folder_path], ("_in.png"))
            targetdata_paths = get_files_from_folders_with_ending([val_folder_path], ("_seg.png"))
        else:
            print(f'Got dir input')
            testdata_paths = get_files_from_folders_with_ending([testdata_path], (".png", ".jpg"))
    else:
        print(f'Got single file input')
        testdata_paths = [testdata_path]
        
    x = np.zeros((len(testdata_paths),) + (size, size) + (3,), dtype="float32")
    for i in range(len(testdata_paths)):
        x[i] = load_img(testdata_paths[i], target_size=(size, size))
    
    cur_conf = load_runconfig(run)
    model = cur_conf.get_model(img_size, num_classes, extra_settings)
    model.load_weights(weights_dir / 'weights')
    model_preds = model.predict(x)
    
    print(test_dir)
    
    if not os.path.exists(test_dir):
            os.makedirs(test_dir)

    for i in range(min(len(model_preds), len(testdata_paths))):
        in_img = ImageOps.autocontrast(array_to_img(x[i]))
        in_img.save(test_dir / f'{i}_input.png')
        
        out_img = ImageOps.autocontrast(array_to_img(model_preds[i]))
        out_img.save(test_dir / f'{i}_network_output.png')

        if len(targetdata_paths) > i:
            target_img = Image.open(targetdata_paths[i]) 
            target_img = target_img.resize((size, size), Image.ANTIALIAS)  
            target_img.save(test_dir / f'{i}_target_output.png')