#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

cd repos

# Yolov5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..

# # Retinanet Model
# git clone https://github.com/fizyr/keras-retinanet
# cd keras-retinanet
# pip install . --user
# cd ..