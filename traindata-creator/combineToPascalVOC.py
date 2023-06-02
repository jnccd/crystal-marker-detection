import argparse
import os
import sys
import time
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog='pascal-voc-compiler', description='Combines multiple dataseries to a pascal voc dataset.')
    parser.add_argument('-tf','--traindata-folders', type=str, help='The folders containing train data, separated by a #.')
    parser.add_argument('-vf','--valdata-folders', type=str, help='The folders containing validation data, separated by a #.')
    args = parser.parse_args()
    
    tf_folders = args.traindata_folders.split('#')
    vf_folders = args.valdata_folders.split('#')
    
    

if __name__ == '__main__':
    main()