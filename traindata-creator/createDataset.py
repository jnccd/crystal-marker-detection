import argparse
import os
import sys
import time
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog='dataset-creator', description='Combines multiple dataseries to a dataset.')
    parser.add_argument('-n','--name', type=str, help='Defines the (folder)name of the dataset.')
#    parser.add_argument('-t','--type', type=str, help='Defines the type of dataset to be build.')
    parser.add_argument('-tf','--traindata-folders', type=str, help='The folders containing train data, separated by a #.')
    parser.add_argument('-vf','--valdata-folders', type=str, help='The folders containing validation data, separated by a #.')
#    parser.add_argument('-r','--ratio', type=str, help='Ratio of traindata to be assigned to valdata.')
    args = parser.parse_args()
    
    tf_folders = args.traindata_folders.split('#')
    vf_folders = args.valdata_folders.split('#')
    
    root_dir = Path(__file__).resolve().parent
    dataset_dir = root_dir / args.name
    
    
    
def get_files_from_folder_with_ending(folders, ending):
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

if __name__ == '__main__':
    main()