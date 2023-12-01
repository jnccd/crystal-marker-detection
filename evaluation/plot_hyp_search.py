import argparse
import ast
import json
import os
import math
import random
from pathlib import Path
import shutil
import sys
import cv2
import re
import scipy
from matplotlib import colors, pyplot as plt, patches as mpatches
from itertools import groupby

import numpy as np

from utility import *

parser = argparse.ArgumentParser(prog='', description='.')
parser.add_argument('-t','--title', type=str, help='Title for the matplotlib plot.')
parser.add_argument('-j','--json-path', type=str, help='Path to the hyperparameter search history json.')
parser.add_argument('-lxp','--label-x-padding', default=18, type=int, help='Label x padding at the start of the plot.')
parser.add_argument('-lap','--label-annotation-padding', default=0.035, type=float, help='Label annotation padding to other labels.')
args = parser.parse_args()

root_dir = Path(__file__).resolve().parent

hyp_history = ast.literal_eval(read_textfile(args.json_path))
hyp_history_scores = [x[0] for x in hyp_history]
hyp_history_highest_scores = [x[1] for x in hyp_history]

def normalize_param_dict(param_dict: dict) -> dict:
    param_dict['ars'] /= 360.0
    param_dict['epochs'] /= 400.0
    param_dict['agns'] /= 256.0
    param_dict['seed'] = 0
    return param_dict

# Crawl hyp_history
max_changes = []
best_score_so_far = None
best_score_params = None
for i in range(len(hyp_history)):
    if best_score_so_far == None:
        best_score_so_far = hyp_history[i][0]
        best_score_params = normalize_param_dict(hyp_history[i][2])
    elif best_score_so_far < hyp_history[i][0]:
        best_score_so_far = hyp_history[i][0]
        new_best_score_params = normalize_param_dict(hyp_history[i][2])
        
        np_new = np.array(list(new_best_score_params.values()))
        np_old = np.array(list(best_score_params.values()))
        np_diff = np.abs(np_new - np_old)
        max_change_index = np.argmax(np_diff)
        max_change_diff = (np_new - np_old)[max_change_index]
        max_change_name = list(new_best_score_params.keys())[max_change_index]
        max_changes.append((max_change_name, max_change_diff, i, hyp_history[i][0]))
        
        best_score_params = new_best_score_params
print(max_changes)

score_color = colors.to_hex((0.15, 0.4, 1))
highest_score_color = colors.to_hex((1, 0.6, 0))

fig, ax = plt.subplots()

ax.plot(range(len(hyp_history_scores)), hyp_history_scores, c = score_color)
ax.plot(range(len(hyp_history_scores)), hyp_history_highest_scores, c = highest_score_color)

cur_y = 0
annot_points = []
annot_padding = args.label_annotation_padding
for max_change in max_changes:
    if cur_y > max_change[3] - annot_padding and cur_x > max_change[2] - 10:
        cur_y = annot_points[-1][1] + annot_padding
        #print(f'move y by {annot_padding}')
    else:
        cur_y = max_change[3]
    cur_x = max_change[2]
    if cur_x < args.label_x_padding:
        cur_x = args.label_x_padding
    print(f'Cur pos: ({cur_x}, {cur_y})')
    annot_points.append((cur_x, cur_y))
    ax.annotate(f'{max_change[0]}: {"+" if max_change[1] > 0 else ""}{round(max_change[1] * 100) }%',
        xy=(cur_x, cur_y),
        xytext=(0, 0),
        textcoords="offset points",
        ha='right', va='bottom')
    cur_y = max_change[3]

# Set labels and layout
ax.set_ylabel('mAP')
ax.set_ylim((0, 1))
ax.set_title(f'VOC 2010 mAP over runs during the Hyperparameter Optimization' if args.title is None else args.title)
ax.set_xlabel('Training Runs')

# Add legend
legend_patches = []
legend_patches.append(mpatches.Patch(color=score_color, label='mAP Score'))
legend_patches.append(mpatches.Patch(color=highest_score_color, label='highest mAP Score'))
ax.legend(handles=legend_patches)

fig.tight_layout()
plt.gcf().set_size_inches(12, 5)
plt.savefig(root_dir / 'plots' / f'{Path(args.json_path).stem}.pdf', dpi=300, bbox_inches='tight')