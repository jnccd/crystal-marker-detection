# Crystal Marker Detection

This is the repo for my Master's Thesis on "ArUco Marker Detection on Photonic Crystals". 
Most of the scripts here assume the one class case for object detection.

## Data Piepline

The scripts in `traindata-creator/` should be used in the following order and generate an object detection dataset in one of the available types.

[arch_data.pdf](https://github.com/jnccd/crystal-marker-detection/files/13716349/arch_data.pdf)

## Training Piepline

Given one or more datasets created in the data pipeline, one of the scripts in `batch-train/` can be invoked to generate training run data.

[arch_train.pdf](https://github.com/jnccd/crystal-marker-detection/files/13716359/arch_train.pdf)
