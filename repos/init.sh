#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# Yolov5
git clone https://github.com/ultralytics/yolov5
cd yolov5
#pip install -r requirements.txt
cd ..

# # Tensorflow Models
# git clone https://github.com/tensorflow/models.git
# cd models
# # TODO: Installing
# cd ..