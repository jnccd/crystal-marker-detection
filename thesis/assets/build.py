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
    plt.gcf().set_size_inches(12, 5)

    plt.savefig(plot_dir / f'{name}.pdf', dpi=300, bbox_inches='tight')


plot_augment_params('best_hyp_run_params', {
    "smart_grid_shuffle_chance": 0.5957674648691555,
    "label_dropout_chance": 0.11627988386742999,
    "perspective_chance": 0.38962228204161486,
    "perspective_strength": 0.053798371264325705,
    "rotation_chance": 0.9574902580520345,
    "rotation_strength": 225.68980145685168,
    "ninety_deg_rotation_chance": 0.3210464611326854,
    "random_crop_chance": 0.19082095693663023,
    "random_crop_v2_chance": 0.2917059831924972,
    "label_move_chance": 0.8596426612346463,
    "label_move_v2_chance": 0.09746777692869586,
    "gauss_noise_chance": 0.8379006635369325,
    "black_dot_chance": 0.8862369463349596,
    "label_curving_chance": 0.6975898700345569
})


