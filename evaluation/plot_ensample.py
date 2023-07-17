import argparse
import json
import os
import math
import random
from pathlib import Path
import shutil
import sys
import cv2
from matplotlib import pyplot as plt

from utils import *

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-rf','--runs-folders', action='append', nargs='+', type=str, help='.')
args = parser.parse_args()

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
    if run_name.endswith('-1'):
        bar_chart_labels.append('-'.join(run_name.split('-')[4:]))
        bar_chart_voc2007_mAPs.append(float(eval_dict['voc2007_mAP']))
        bar_chart_voc2010_mAPs.append(float(eval_dict['voc2010_mAP']))
        bar_chart_coco_mAPs.append(float(eval_dict['coco_mAP']))
        
bar_chart_voc2007_mAPs = [round(x, 3) for x in bar_chart_voc2007_mAPs]
bar_chart_voc2010_mAPs = [round(x, 3) for x in bar_chart_voc2010_mAPs]
bar_chart_coco_mAPs = [round(x, 3) for x in bar_chart_coco_mAPs]
        
# Create barchart
x = np.arange(len(bar_chart_labels))
width = 0.35 / 3
fig, ax = plt.subplots()
v7_bars = ax.bar(x - width, bar_chart_voc2007_mAPs, width, label='voc2007 mAP')
v10_bars = ax.bar(x, bar_chart_voc2010_mAPs, width, label='voc2010 mAP')
coco_bars = ax.bar(x + width, bar_chart_coco_mAPs, width, label='coco mAP')
ax.set_ylabel('mAP')
ax.set_title('mAP per run')
ax.set_xticks(x)
ax.set_xticklabels(bar_chart_labels, rotation=30, ha='right')
ax.legend()
autolabel(v7_bars, ax)
autolabel(v10_bars, ax)
autolabel(coco_bars, ax)
fig.tight_layout()
plt.show()