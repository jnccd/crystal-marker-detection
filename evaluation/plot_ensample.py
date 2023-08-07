import argparse
import json
import os
import math
import random
from pathlib import Path
import shutil
import sys
import cv2
import re
from matplotlib import colors, pyplot as plt
from itertools import groupby

import numpy as np

from utility import *

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-n','--name', type=str, help='.')
parser.add_argument('-t','--title', type=str, help='.')
parser.add_argument('-rf','--runs-folders', action='append', nargs='+', type=str, help='.')
parser.add_argument('-rnp','--run-name-pattern', type=str, help='Regex filter for run name.')
parser.add_argument('-pi','--part-index', type=int, help='Index of the part number in the run name, split by "-", if set runs are grouped by the .')
parser.add_argument('-ci','--config-index', type=int, help='Index of the config number in the run name, split by "-", if set runs are grouped by the .')
args = parser.parse_args()

name_pattern = re.compile(args.run_name_pattern) if args.run_name_pattern is not None else None

root_dir = Path(__file__).resolve().parent
runs_folders = flatten(args.runs_folders)
eval_paths = flatten([[ x for x in Path(run_folders).glob('**/evals.json') 
                        if not str(x).__contains__("_old")] 
                        for run_folders in runs_folders])

# Group eval_paths by part_index if it is set, else use one elem lists
if args.part_index is not None:
    eval_paths_keys = []
    for x in eval_paths:
        run_name = x.parent.parent.parent.stem
        run_name_split = run_name.split('-')
        run_name_split.pop(args.part_index)
        eval_paths_keys.append(('-'.join(run_name_split), x))
    eval_paths_grouped = [[y for y in eval_paths_keys if y[0]==x] for x in set(map(lambda x: x[0], eval_paths_keys))]
else:
    eval_paths_keys = [x.parent.parent.parent.stem for x in eval_paths]
    eval_paths_grouped = [[(x, y)] for x, y in zip(eval_paths_keys, eval_paths)]
# print(eval_paths_grouped)
# print(eval_paths_grouped[0])
# sys.exit(0)

# Create Eval database
bar_chart_entries = []
for eval_paths_group in eval_paths_grouped:
    bar_chart_entry = {}
    
    group_voc2007_mAPs = []
    group_voc2010_mAPs = []
    group_coco_mAPs = []
    
    for eval_name, eval_path in eval_paths_group:
        train_def_path = eval_path.parent.parent / 'training-def.json'
        
        if not train_def_path.is_file():
            print(f'Couldnt find file of {eval_name}, {eval_path}')
            continue
        
        eval_dict = json.loads(read_textfile(eval_path).replace("    ", "").replace("\n", ""))
        train_def_dict = json.loads(read_textfile(train_def_path).replace("    ", "").replace("\n", ""))
        
        run_name: str = train_def_dict['run_name']
        if name_pattern is not None and not name_pattern.match(run_name):
            continue
        
        group_voc2007_mAPs.append(float(eval_dict['voc2007_mAP']))
        group_voc2010_mAPs.append(float(eval_dict['voc2010_mAP']))
        group_coco_mAPs.append(float(eval_dict['coco_mAP']))
        
    if group_voc2007_mAPs == []:
        continue
    
    bar_chart_entry['label'] = '-'.join(eval_name.split('-')[2:])
    bar_chart_entry['run_name'] = run_name
    if args.config_index is not None:
        bar_chart_entry['config'] = run_name.split('-')[args.config_index]
        bar_chart_entry['label'] = f'{float(bar_chart_entry["config"])}%'
    
    bar_chart_entry['voc2007_mAP'] = np.mean(group_voc2007_mAPs) if len(group_voc2007_mAPs) > 0 else 0
    bar_chart_entry['voc2010_mAP'] = np.mean(group_voc2010_mAPs) if len(group_voc2010_mAPs) > 0 else 0
    bar_chart_entry['coco_mAP'] = np.mean(group_coco_mAPs) if len(group_coco_mAPs) > 0 else 0
    
    bar_chart_entry['voc2007_mAP_error'] = np.std(group_voc2007_mAPs) if len(group_voc2007_mAPs) > 0 else 0
    bar_chart_entry['voc2010_mAP_errors'] = np.std(group_voc2010_mAPs) if len(group_voc2010_mAPs) > 0 else 0
    bar_chart_entry['coco_mAP_errors'] = np.std(group_coco_mAPs) if len(group_coco_mAPs) > 0 else 0
    
    bar_chart_entries.append(bar_chart_entry)

# print('bar_chart_voc2007_mAPs', [x['voc2007_mAP'] for x in bar_chart_entries])
# print('bar_chart_voc2010_mAPs', [x['voc2010_mAP'] for x in bar_chart_entries])
# print('bar_chart_coco_mAPs', [x['coco_mAP'] for x in bar_chart_entries])
# print('bar_chart_voc2007_mAP_errors', [x['voc2007_mAP_error'] for x in bar_chart_entries])
# print('bar_chart_coco_mAP_errors', [x['coco_mAP_errors'] for x in bar_chart_entries])

if args.config_index is not None:
    bar_chart_entries.sort(key=lambda x: x['config'])
print('bar_chart_entries', bar_chart_entries)

# Create barchart
x = np.arange(len(bar_chart_entries))
width = 0.6 / 3
fig, ax = plt.subplots()
v7_bars = ax.bar(x - width, [x['voc2007_mAP'] for x in bar_chart_entries], width, yerr=[x['voc2007_mAP_error'] for x in bar_chart_entries], label='voc2007 mAP', color=colors.to_hex((0.15, 0.4, 1)))
v10_bars = ax.bar(x, [x['voc2010_mAP'] for x in bar_chart_entries], width, yerr=[x['voc2010_mAP_errors'] for x in bar_chart_entries], label='voc2010 mAP', color=colors.to_hex((0.3, 0.65, 1)))
coco_bars = ax.bar(x + width, [x['coco_mAP'] for x in bar_chart_entries], width, yerr=[x['coco_mAP_errors'] for x in bar_chart_entries], label='coco mAP', color=colors.to_hex((1, 0.6, 0)))
ax.set_ylim((0, max(np.max([x['voc2007_mAP'] for x in bar_chart_entries]), np.max([x['voc2010_mAP'] for x in bar_chart_entries]), np.max([x['coco_mAP'] for x in bar_chart_entries])) * 1.1))
ax.set_ylabel('mAP')
ax.set_title(f'mAP per run in {args.name.replace("-", " ")}' if args.title is None else args.title)
ax.set_xticks(x)
ax.set_xticklabels([x['label'] for x in bar_chart_entries], rotation=30, ha='right')
ax.legend()
autolabel(v7_bars, ax)
autolabel(v10_bars, ax)
autolabel(coco_bars, ax)
fig.tight_layout()
plt.gcf().set_size_inches(20, 9)
plt.savefig(root_dir / f'{args.name}.pdf', dpi=300)