import argparse
import ast
import json
import math
import cv2
from pathlib import Path
from shapely import Polygon

from utility import *

# Parse
parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-t','--testseries-path', type=str, default='', help='.')
parser.add_argument('-i','--images-folder', type=str, default='', help='.')
args = parser.parse_args()

# --- Load paths
dataseries_path = Path('traindata-creator/dataseries')
# Input human results
root_dir = Path(__file__).resolve().parent
results_dir = root_dir / 'human-testing/results'
result_paths = [Path(x) for x in get_files_from_folders_with_ending([results_dir], (".json"))]
# Load testseries
testseries_path = Path(args.testseries_path)
testseries_imgs = [cv2.imread(str(x)) for x in get_files_from_folders_with_ending([testseries_path], '_in.png')]
testseries_polys = [[Polygon(b) for b in ast.literal_eval(read_textfile(p))] for p in get_files_from_folders_with_ending([testseries_path], '_vertices.txt')]

for dict_path in result_paths:
    out_dir = create_dir_if_not_exists(results_dir / dict_path.stem)
    print(f'Bulding {out_dir}...')
    
    os.system(f'python traindata-creator/createCocoJsonDataseries.py -i {args.images_folder} -c {dict_path}')
    dict_dataseries_path = dataseries_path / f'coco-{dict_path.stem}'
    
    dict_dataseries_imgs = [cv2.imread(str(x)) for x in get_files_from_folders_with_ending([dict_dataseries_path], '_in.png')]
    dict_dataseries_polys = [[Polygon(b) for b in ast.literal_eval(read_textfile(p))] for p in get_files_from_folders_with_ending([dict_dataseries_path], '_vertices.txt')]
    
    for i in range(len(testseries_imgs)):
        img = dict_dataseries_imgs[i]
        polys = dict_dataseries_polys[i]
        img_h, img_w = img.shape[:2]
        
        test_img = testseries_imgs[i]
        test_polys = testseries_polys[i]
        test_img_h, test_img_w = test_img.shape[:2]
        
        bboxes = [x.bounds for x in polys]
        bboxes = [(x * test_img_w / img_w, # Rescale for test img
                   y * test_img_h / img_h, 
                   mx * test_img_w / img_w, 
                   my * test_img_h / img_h) for (x, y, mx, my) in bboxes]
        
        test_bboxes = [x.bounds for x in test_polys]
        
        cv2.imwrite(str(out_dir / f'{i}_input.png'), test_img)
        write_textfile('\n'.join([f'{x} {y} {mx} {my} 1' for (x, y, mx, my) in bboxes]), str(out_dir / f'{i}_network_output.txt'))
        write_textfile('\n'.join([f'{x} {y} {mx} {my}' for (x, y, mx, my) in test_bboxes]), str(out_dir / f'{i}_target_output.txt'))
        
    os.system(f'python evaluation/analyze.py -av {out_dir}')