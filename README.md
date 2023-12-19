# Crystal Marker Detection

This is the repo for my Master's Thesis on "ArUco Marker Detection on Photonic Crystals". 
Most of the scripts here assume the one class case for object detection.

## Data Piepline

The scripts in `traindata-creator/` should be used in the following order and generate an object detection dataset in one of the available types.

![Screenshot_9223371259636314352_ScreenshotTool](https://github.com/jnccd/crystal-marker-detection/assets/19777592/3fff2633-8b18-4581-a7ef-f8a06381a012)

## Training Piepline

Given one or more datasets created in the data pipeline, one of the scripts in `batch-train/` can be invoked to generate training run data.

![Screenshot_9223371259636306634_ScreenshotTool](https://github.com/jnccd/crystal-marker-detection/assets/19777592/a9b65c91-d054-4d4b-ab2d-0e20dd6164b1)
