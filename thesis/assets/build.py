import os
import shutil
import numpy as np
from pathlib import Path
from graphviz import Digraph, Graph, Source
import matplotlib.pyplot as plt

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

root_dir = Path(__file__).resolve().parent
graph_dir = root_dir / 'graph'
dot_files = list(graph_dir.glob('*.dot'))

print(f'building {len(dot_files)} graphs...')

for dot_file in dot_files:
    ps = Source(read_textfile(dot_file), format='pdf')
    ps.render(str(graph_dir / dot_file.stem))
    os.system(f'rm {graph_dir / dot_file.stem}')
    