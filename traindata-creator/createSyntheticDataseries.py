import cv2
from utils import *
from pathlib import Path

root_dir = Path(__file__).resolve().parent
assets_dir = root_dir / 'raw/synthetic-builder-assets'
foreground_textures_dir = assets_dir / 'foreground_textures'
background_textures_dir = assets_dir / 'background_textures'

fore_textures = [cv2.imread(str(p)) for p in get_files_from_folders_with_ending([foreground_textures_dir], '.png')]
back_textures = [cv2.imread(str(p)) for p in get_files_from_folders_with_ending([background_textures_dir], '.png')]

