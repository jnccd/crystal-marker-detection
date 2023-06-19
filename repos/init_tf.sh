#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# Retinanet Model
git clone https://github.com/fizyr/keras-retinanet
cd keras-retinanet
pip install . --user
cd ..

# # Tensorflow Models
# git clone https://github.com/tensorflow/models.git
# cd models
# # TODO: Installing
# cd ..