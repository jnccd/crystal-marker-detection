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

def test(
    testdata,
    run: str = 'default', 
    size: int = 320, 
    ):
    
    print(f'Testing {run}...')
    
    root_dir = Path(__file__).resolve().parent
    runs_dir = root_dir / 'runs'
    run_dir = runs_dir / f'run-{run}'
    test_dir = run_dir / 'test'
    weights_dir = run_dir / 'weights'
    
    testdata_path = Path(testdata)
    if testdata_path.is_dir():
        print(f'Got dir input')
        testdata_paths = get_files_from_folders_with_ending([testdata_path], (".png", ".jpg"))
    else:
        print(f'Got single file input')
        testdata_paths = [testdata_path]
    x = np.zeros((len(testdata_paths),) + (size, size) + (3,), dtype="float32")
    for testdata_p in testdata_paths:
        x[0] = load_img(testdata_p, target_size=(size, size))
    
    cur_conf = load_runconfig(run)
    model = cur_conf.model
    model.load_weights(weights_dir / 'weights')
    model_preds = model.predict(x)
    
    print(test_dir)
    
    if not os.path.exists(test_dir):
            os.makedirs(test_dir)

    for i in range(0, min(len(model_preds), len(testdata_paths))):
        in_img = ImageOps.autocontrast(array_to_img(x[i]))
        in_img.save(test_dir / f'{i}_input.png')
        
        out_img = ImageOps.autocontrast(array_to_img(model_preds[i]))
        out_img.save(test_dir / f'{i}_network_output.png')
    