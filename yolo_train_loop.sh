#!/usr/bin/env bash

run_name=test_sh_w_hyp
dataset_name=yolov5-640-aug-good-pics-v2
valset_name=yolov5-640-on-skin-valset

python repos/yolov5/train.py --name $run_name --img 640 --batch 8 --epochs 300 --data traindata-creator/dataset/$dataset_name/$dataset_name.yaml --weights yolov5s.pt #--hyp hyp.no-augmentation.yaml 
rm ./yolov5s.pt
python repos/yolo_gen_evaldata.py -r $run_name -df traindata-creator/dataset/$valset_name/
python -m cmd_tf -av repos/evaldata/yolov5-$run_name/