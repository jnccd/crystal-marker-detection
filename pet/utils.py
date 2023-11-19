import math
import os
from pathlib import Path
import shutil
import numpy as np
from torch import Tensor
from shapely import LineString

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

def write_textfile(text, tf_path):
    with open(tf_path, "w") as text_file:
        text_file.write(text)
        
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

def flatten(list):
    return [item for sublist in list for item in sublist]

def unflatten(list, chunk_size):
    return [list[n:n+chunk_size] for n in range(0, len(list), chunk_size)]

# Horrible function implemented shortly before october
def name_of_object(arg):
    try:
        return arg.__name__
    except AttributeError:
        pass

    for name, value in globals().items():
        if value is arg and not name.startswith('_'):
            return name
        
# math
        
def eelongate(l: LineString, mult: float):
    x, y = l.xy
    x_diff = x[1] - x[0]
    y_diff = y[1] - y[0]
    return LineString([(x[0] - x_diff * mult, y[0] - y_diff * mult), (x[1] + x_diff * mult, y[1] + y_diff * mult)])

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def add(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])

def diff(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])

def scalar_mult(p, s):
    return (p[0] * s, p[1] * s)

def dot_product(p1, p2):
    return p1[0] * p2[0] + p1[1] * p2[1]       

# mean Average Point Distance
def mAPD_2D(preds_batch: Tensor, gts_batch: Tensor, max_possible_distance: float = 226) -> float:
    
    mPDs = []
    #print(preds_batch, gts_batch)
    
    for preds, gts in zip(preds_batch, gts_batch):
        mPD_table = []
        
        for j, pred in enumerate(preds):
            best_fits = [[distance(x, pred), i] for i, x in enumerate(gts)]
            best_fits = sorted(best_fits, key=lambda x: x[0])
            best_fit = best_fits[0] # [distance, i(gt)]
            
            mPD_table.append(best_fit)
            
        mPD_table = sorted(mPD_table, key=lambda x: x[0])
        #print(mPD_table)
        
        distances = []
        used_gt_indices = []
        for entry in mPD_table:
            if not entry[1] in used_gt_indices:
                distances.append(entry[0])
                used_gt_indices.append(entry[1])
            else:
                distances.append(max_possible_distance)
        
        mPDs.append(np.mean(distances))
        
    return np.mean(mPDs)