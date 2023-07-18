import argparse
import json
import os
import math
import random
from pathlib import Path
import shutil
import sys
import cv2
from matplotlib import colors, pyplot as plt

from utils import *

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-n','--name', type=str, help='.')
parser.add_argument('-rf','--runs-folders', action='append', nargs='+', type=str, help='.')
parser.add_argument('-rns','--run-name-suffix', type=str, help='.')
args = parser.parse_args()

root_dir = Path(__file__).resolve().parent
runs_folders = flatten(args.runs_folders)
eval_paths = flatten([[ x for x in Path(run_folders).glob('**/evals.json') 
                        if not str(x).__contains__("_old")] for run_folders in runs_folders])
#print(eval_paths)

bar_chart_labels = []
bar_chart_voc2007_mAPs = []
bar_chart_voc2010_mAPs = []
bar_chart_coco_mAPs = []

for eval_path in eval_paths:
    train_def_path = eval_path.parent.parent / 'training-def.json'
    #print(train_def_path)
    
    if not train_def_path.is_file():
        continue
    
    eval_dict = json.loads(read_textfile(eval_path).replace(" ", "").replace("\n", ""))
    train_def_dict = json.loads(read_textfile(train_def_path).replace(" ", "").replace("\n", ""))
    
    run_name: str = train_def_dict['run_name']
    
    if args.run_name_suffix is not None and not run_name.endswith(args.run_name_suffix):
        continue
        
    bar_chart_labels.append('-'.join(run_name.split('-')[4:-1]))
    bar_chart_voc2007_mAPs.append(float(eval_dict['voc2007_mAP']))
    bar_chart_voc2010_mAPs.append(float(eval_dict['voc2010_mAP']))
    bar_chart_coco_mAPs.append(float(eval_dict['coco_mAP']))
        
bar_chart_voc2007_mAPs = [round(x, 3) for x in bar_chart_voc2007_mAPs]
bar_chart_voc2010_mAPs = [round(x, 3) for x in bar_chart_voc2010_mAPs]
bar_chart_coco_mAPs = [round(x, 3) for x in bar_chart_coco_mAPs]

#print('debug', list(zip(bar_chart_labels, bar_chart_coco_mAPs)))

# Create barchart
x = np.arange(len(bar_chart_labels))
width = 0.6 / 3
fig, ax = plt.subplots()
v7_bars = ax.bar(x - width, bar_chart_voc2007_mAPs, width, label='voc2007 mAP', color=colors.to_hex((0.15, 0.4, 1)))
v10_bars = ax.bar(x, bar_chart_voc2010_mAPs, width, label='voc2010 mAP', color=colors.to_hex((0.3, 0.65, 1)))
coco_bars = ax.bar(x + width, bar_chart_coco_mAPs, width, label='coco mAP', color=colors.to_hex((1, 0.6, 0)))
ax.set_ylabel('mAP')
ax.set_title(f'mAP per run in {args.name.replace("-", " ")}')
ax.set_xticks(x)
ax.set_xticklabels(bar_chart_labels, rotation=30, ha='right')
ax.legend()
autolabel(v7_bars, ax)
autolabel(v10_bars, ax)
autolabel(coco_bars, ax)
fig.tight_layout()
plt.gcf().set_size_inches(20, 9)
plt.savefig(root_dir / f'{args.name}.pdf', dpi=300)