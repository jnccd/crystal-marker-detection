import argparse
import json
import math
import cv2
from utils import *
from pathlib import Path

parser = argparse.ArgumentParser(prog='coco-json-dataseries-importer', description='Creates dataseries from coco json marked traindata.')
parser.add_argument('-i','--images-folder', type=str, help='The path to the folder containing an image series.')
parser.add_argument('-c','--coco-json', type=str, help='The path to the folder containing an image series.')
args = parser.parse_args()

# Load paths
root_dir = Path(__file__).resolve().parent
input_dir = Path(args.images_folder)
dataseries_dir = create_dir_if_not_exists(root_dir / 'dataseries' / f'coco-{Path(args.coco_json).stem}', clear=True)
img_paths = [Path(x) for x in get_files_from_folders_with_ending([args.images_folder], (".png", ".jpg"))]
coco_dict = json.loads(read_textfile(args.coco_json).replace("    ", "").replace("\n", ""))

# Convert
dict_annots = coco_dict['annotations']
dict_images = coco_dict['images']
for i, img_path in enumerate(img_paths):
    dict_images_hits = list(filter(lambda x: x['file_name'] == img_path.name, dict_images))
    img = cv2.imread(str(img_path))
    img_vertecies = []
    
    # Some images may not have entries
    if len(dict_images_hits) > 0:
        dict_img = dict_images_hits[0]
        img_id = dict_img['id']
        img_annots = list(filter(lambda x: x['image_id'] == img_id, dict_annots))
        
        # Build int tuple vertex list
        for img_annot in img_annots:
            for segmentation in img_annot['segmentation']:
                img_vertecies.append([tuple(x) for x in unflatten([int(y) for y in segmentation], 2)])
    
    # Write out
    cv2.imwrite(str(dataseries_dir / f'{img_path.stem}_in.png'), img)
    write_textfile(str(img_vertecies), dataseries_dir / f'{img_path.stem}_vertices.txt')
    