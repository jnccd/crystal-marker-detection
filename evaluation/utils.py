import os
from pathlib import Path
import shutil
from typing import List

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# String stuff
def startswith_any(filename: str, prefs: List[str]):
    for pref in prefs:
        if filename.startswith(pref):
            return True
    return False

# Paths
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

# Other
def flatten(list):
    return [item for sublist in list for item in sublist]
    
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# --- Textfiles -------------------------------------------------------------------------------------------------------------------------

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

def write_textfile(text, tf_path):
    with open(tf_path, "w") as text_file:
        text_file.write(text)