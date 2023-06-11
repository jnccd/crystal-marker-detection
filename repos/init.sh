#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) 
echo 'Hi, Im in '$SCRIPT_DIR

cd $SCRIPT_DIR

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

cd ..