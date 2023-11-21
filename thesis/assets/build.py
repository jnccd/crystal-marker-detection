import os
import shutil
import numpy as np
from pathlib import Path
from graphviz import Digraph, Graph, Source
from matplotlib import colors, pyplot as plt

# Utility

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

def create_dir_if_not_exists(dir: Path, clear = False):
    if clear and os.path.isdir(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# ---

root_dir = Path(__file__).resolve().parent
graph_dir = root_dir / 'graph'
dot_files = list(graph_dir.glob('*.dot'))

print(f'building {len(dot_files)} graphs...')

for dot_file in dot_files:
    ps = Source(read_textfile(dot_file), format='pdf')
    ps.render(str(graph_dir / dot_file.stem))
    os.system(f'rm {graph_dir / dot_file.stem}')
    
# ---

print(f'building plots...')

plot_dir = create_dir_if_not_exists(root_dir / 'plot')

def plot_augment_params(name: str, params: dict):
    bar_x = np.arange(len(params.keys()))
    bar_y = [a for a in params.values()]

    # Get values to be between 0 and 100
    for i in range(len(bar_y)):
        if [a for a in params.keys()][i] == "rotation_strength":
            bar_y[i] = (bar_y[i] * 100) / 360
        else:
            bar_y[i] *= 100

    fig, ax = plt.subplots()

    ax.bar(
        x=      bar_x, 
        height= bar_y, 
        color=  [colors.to_hex((1, 0.6, 0)) if x.endswith('_strength') else colors.to_hex((0.15, 0.4, 1)) for x in params.keys()]
        )

    ax.set_ylim((0, 100))
    ax.set_ylabel('Chance in % or maximum strength in %')
    ax.set_title(f'Augmentation Parameters')
    ax.set_xticks(bar_x)
    ax.set_xlabel('Augmentations')
    ax.set_xticklabels([x.replace("_", " ") for x in params.keys()], rotation=30, ha='right')

    fig.tight_layout()
    plt.gcf().set_size_inches(20, 9)

    plt.savefig(plot_dir / f'{name}.pdf', dpi=300)


plot_augment_params('test', {
    "smart_grid_shuffle_chance": 0.16536455065004735,
    "label_dropout_chance": 0.3406092762590903,
    "perspective_chance": 0.9026868390392013,
    "perspective_strength": 0.44744759491769104,
    "rotation_chance": 0.8397534486075489,
    "rotation_strength": 269.5297433583759,
    "ninety_deg_rotation_chance": 0.4779575209987885,
    "random_crop_chance": 0.07811140975400804,
    "random_crop_v2_chance": 0.09484458554495206,
    "label_move_chance": 0.3281069403856025,
    "label_move_v2_chance": 0.04946212899649677,
    "gauss_noise_chance": 0.2850102716939429,
    "black_dot_chance": 0.5932304638260782,
    "label_curving_chance": 0.09002886339102176
})


