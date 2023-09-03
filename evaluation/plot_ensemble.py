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
parser.add_argument('-r','--runs-folders', action='append', nargs='+', type=str, help='.')
parser.add_argument('-rnp','--run-name-pattern', type=str, help='Regex filter for run name.')
parser.add_argument('-pi','--part-index', type=int, help='Index of the part number in the run name, split by "-", if set runs are grouped by the.')
parser.add_argument('-ci','--config-index', type=int, help='Index of the config number in the run name, split by "-", if set runs are grouped by the.')
parser.add_argument('-cu','--config-unit', type=str, help='How should the config be understood? "%", "10%", "deg".')
parser.add_argument('-bfl','--best-fit-lines', action='store_true', help='Adds a degree 1 best fit line over the data.')
args = parser.parse_args()

name_pattern = re.compile(args.run_name_pattern) if args.run_name_pattern is not None else None

root_dir = Path(__file__).resolve().parent
runs_paths = get_all_subfolder_run_dirs(flatten(args.runs_folders))

# Group eval_paths by part_index if it is set, else use one elem lists
if args.part_index is not None:
    run_paths_keys = []
    for x in runs_paths:
        run_name = x['run_root'].stem
        run_name_split = run_name.split('-')
        run_name_split.pop(args.part_index)
        run_paths_keys.append(('-'.join(run_name_split), x))
    runs_paths_grouped = [[y for y in run_paths_keys if y[0]==x] for x in set(map(lambda x: x[0], run_paths_keys))]
else:
    run_paths_keys = [x['run_root'].stem for x in runs_paths]
    runs_paths_grouped = [[(x, y)] for x, y in zip(run_paths_keys, runs_paths)]
# print(eval_paths_grouped)
# print(eval_paths_grouped[0])
# sys.exit(0)

# Create Eval database
bar_chart_entries = []
for runs_paths_group in runs_paths_grouped:
    bar_chart_entry = {}
    
    group_voc2007_mAPs = []
    group_voc2010_mAPs = []
    group_coco_mAPs = []
    
    for run_paths_name, run_paths in runs_paths_group:
        eval_dict = json.loads(read_textfile(run_paths['eval']).replace("    ", "").replace("\n", ""))
        train_def_dict = json.loads(read_textfile(run_paths['train_def']).replace("    ", "").replace("\n", ""))
        
        run_name: str = train_def_dict['run_name']
        if name_pattern is not None and not name_pattern.match(run_name):
            continue
        
        group_voc2007_mAPs.append(float(eval_dict['voc2007_mAP']))
        group_voc2010_mAPs.append(float(eval_dict['voc2010_mAP']))
        group_coco_mAPs.append(float(eval_dict['coco_mAP']))
        
    if group_voc2007_mAPs == []:
        continue
    
    bar_chart_entry['label'] = '-'.join(run_paths_name.split('-')[2:])
    bar_chart_entry['run_name'] = run_name
    if args.config_index is not None:
        bar_chart_entry['config'] = run_name.split('-')[args.config_index]
        
        if args.config_unit is None:
            bar_chart_entry['label'] = bar_chart_entry["config"]
        elif args.config_unit == '%':
            bar_chart_entry['label'] = f'{float(bar_chart_entry["config"])}%'
        elif args.config_unit == '10%':
            bar_chart_entry['label'] = f'{str(bar_chart_entry["config"]).lstrip("0")}0%'
        elif args.config_unit == 'deg%':
            bar_chart_entry['label'] = f'{float(bar_chart_entry["config"])}°'
    
    bar_chart_entry['voc2007_mAP'] = np.mean(group_voc2007_mAPs) if len(group_voc2007_mAPs) > 0 else 0
    bar_chart_entry['voc2010_mAP'] = np.mean(group_voc2010_mAPs) if len(group_voc2010_mAPs) > 0 else 0
    bar_chart_entry['coco_mAP'] = np.mean(group_coco_mAPs) if len(group_coco_mAPs) > 0 else 0
    
    bar_chart_entry['voc2007_mAP_error'] = np.std(group_voc2007_mAPs) if len(group_voc2007_mAPs) > 0 else 0
    bar_chart_entry['voc2010_mAP_error'] = np.std(group_voc2010_mAPs) if len(group_voc2010_mAPs) > 0 else 0
    bar_chart_entry['coco_mAP_error'] = np.std(group_coco_mAPs) if len(group_coco_mAPs) > 0 else 0
    
    bar_chart_entries.append(bar_chart_entry)

# print('bar_chart_voc2007_mAPs', [x['voc2007_mAP'] for x in bar_chart_entries])
# print('bar_chart_voc2010_mAPs', [x['voc2010_mAP'] for x in bar_chart_entries])
# print('bar_chart_coco_mAPs', [x['coco_mAP'] for x in bar_chart_entries])
# print('bar_chart_voc2007_mAP_errors', [x['voc2007_mAP_error'] for x in bar_chart_entries])
# print('bar_chart_coco_mAP_errors', [x['coco_mAP_errors'] for x in bar_chart_entries])

if args.config_index is not None:
    bar_chart_entries.sort(key=lambda x: float(x['config']))
else:
    bar_chart_entries.sort(key=lambda x: x['label'])

# --- Create barchart
x = np.arange(len(bar_chart_entries))
width = 0.6 / 3
fig, ax = plt.subplots()
data_lines = ['voc2007_mAP', 'voc2010_mAP', 'coco_mAP']
data_colors = {
    'voc2007_mAP': colors.to_hex((0.15, 0.4, 1)),
    'voc2010_mAP': colors.to_hex((0.3, 0.65, 1)),
    'coco_mAP': colors.to_hex((1, 0.6, 0)),
}
data_lines_x_offset = {
    'voc2007_mAP': x - width,
    'voc2010_mAP': x,
    'coco_mAP': x + width,
}

# Add data bars
bars = []
for data_line in data_lines:
    bars.append(
        ax.bar(
            x=      data_lines_x_offset[data_line], 
            height= [x[data_line] for x in bar_chart_entries], 
            width=  width, 
            yerr=   [x[f'{data_line}_error'] for x in bar_chart_entries], 
            label=  data_line.replace('_', ' '), 
            color=  data_colors[data_line]
            )
        )

# Add best fit line
if args.best_fit_lines is not None:
    for data_line in data_lines:
        theta = np.polyfit(x, [x[data_line] for x in bar_chart_entries], 1)
        y_line = theta[1] + theta[0] * x
        plt.plot(data_lines_x_offset[data_line], y_line, data_colors[data_line])
        ax.annotate(str(round(theta[0] * 10000, 2)),
            xy=(data_lines_x_offset[data_line][-1], y_line[-1]),
            xytext=(8, -3),
            textcoords="offset points",
            ha='left', va='bottom')

# Set labels and layout
ax.set_ylim((0, max([np.max([x[data_line] for x in bar_chart_entries]) for data_line in data_lines]) * 1.1))
ax.set_ylabel('mAP')
ax.set_title(f'mAP per run in {args.name.replace("-", " ")}' if args.title is None else args.title)
ax.set_xticks(x)
ax.set_xticklabels([x['label'] for x in bar_chart_entries], rotation=30, ha='right')
ax.legend()
for bar in bars:
    autolabel(bar, ax)
fig.tight_layout()
plt.gcf().set_size_inches(20, 9)

plt.savefig(root_dir / 'plots' / f'{args.name}.pdf', dpi=300)