import os
from pathlib import Path
import re
import shutil
from typing import List

yolov5_pattern = re.compile('yolov5(.?)$')
yolov8_pattern = re.compile('yolov8(.?)$|yolov5(.?)u$')

# String stuff
def startswith_any(filename: str, prefs: List[str]):
    for pref in prefs:
        if filename.startswith(pref):
            return True
    return False

# Paths
def get_files_from_folders_with_ending(folders, ending):
    paths = []
    for folder in folders:
        paths.extend(sorted(
            [
                os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(ending)
            ]
        ))
    return paths
def create_dir_if_not_exists(dir: Path, clear = False):
    if clear and os.path.isdir(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_all_subfolder_run_dirs(search_root_dirs):
    """Finds all subfolders that are run folders as well as paths to important files within.

    Arguments:
      search_root_dirs: A list of paths to search all subfolders of for run files.

    Returns:
      A list of dicts for all found run folders [{'run_root', 'train_def', 'eval', 'tests': {'$test_name': {'test_def', 'eval'} } }].
      (The leafs are omitted)
    """
    
    train_def_paths = flatten([[x for x in Path(search_root_dir).glob('**/training-def.json')
                                if not str(x).__contains__("_old")] 
                                for search_root_dir in search_root_dirs])
    run_paths = [x.parent.parent if x.parent.stem == 'test' else x.parent 
                    for x in train_def_paths]
    eval_paths = [x / 'test/evals/evals.json' for x in run_paths]
    test_defs_paths = [x.glob('**/_test-def.json') for x in run_paths]
    tests_paths = [[(x.parent.stem, {'test_def': x, 'eval': x.parent / 'evals/evals.json'}) for x in test_def_paths if os.path.isfile(x.parent / 'evals/evals.json')] for test_def_paths in test_defs_paths]
    tests_path_dicts = [dict(x) if x != [] else {} for x in tests_paths]
    
    tuple_paths_list = list(zip(run_paths, train_def_paths, eval_paths, tests_path_dicts))
    dict_keys = ['run_root', 'train_def', 'eval', 'tests']
    dict_list = [dict(zip(dict_keys, x)) for x in tuple_paths_list 
                    if os.path.exists(x[2]) and os.path.isfile(x[2]) and os.path.exists(x[1]) and os.path.isfile(x[1])]
    dict_list.sort(key=lambda d: d['run_root'].stem)
    return dict_list

# Other
def flatten(list):
    return [item for sublist in list for item in sublist]

def unflatten(list, chunk_size):
    return [list[n:n+chunk_size] for n in range(0, len(list), chunk_size)]
    
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 3)
        if height > 0:
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

# --- Textfiles -------------------------------------------------------------------------------------------------------------------------

def read_textfile(tf_path):
    with open(tf_path, 'r') as file:
        file_text = file.read()
    return file_text

def write_textfile(text, tf_path):
    with open(tf_path, "w") as text_file:
        text_file.write(text)