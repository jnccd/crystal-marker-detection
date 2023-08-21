import os
from pathlib import Path
import shutil

import cv2

# --- Textfiles -------------------------------------------------------------------------------------------------------------------------

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

def write_textfile(text, tf_path):
    with open(tf_path, "w") as text_file:
        text_file.write(text)
        
# --- Other -------------------------------------------------------------------------------------------------------------------------

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

def set_img_width(img, max_width):
    img_h, img_w = img.shape[:2]
    resize_factor = float(max_width) / img_w
    target_size = (max_width, int(img_h * resize_factor))
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def set_img_height(img, max_height):
    img_h, img_w = img.shape[:2]
    resize_factor = float(max_height) / img_h
    target_size = (int(img_w * resize_factor), max_height)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def create_dir_if_not_exists(dir: Path, clear = False):
    if clear and os.path.isdir(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def flatten(list):
    return [item for sublist in list for item in sublist]