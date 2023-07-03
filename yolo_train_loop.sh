#!/usr/bin/env bash

run_name=test_whyp
dataset_name=yolov5-640-aug-good-pics-v2
valset_name=yolov5-640-on-skin-valset

# Apply start args
if [ "$1" != '' ]; then
    run_name=$1
	echo "Set runname to $run_name!"
fi
if [ "$2" != '' ]; then
    dataset_name=$2
	echo "Set dataset_name to $dataset_name!"
fi

# Set parameter args
if [[ $run_name == *whyp ]]; then
    hyp=''
else
    hyp='--hyp hyp.no-augmentation.yaml'
fi
echo "hyp is '$hyp'!"

python repos/yolov5/train.py --name $run_name --img 640 --batch 8 --epochs 300 --data traindata-creator/dataset/$dataset_name/$dataset_name.yaml --weights yolov5s.pt $hyp
rm ./yolov5s.pt
python repos/yolo_gen_evaldata.py -r $run_name -df traindata-creator/dataset/$valset_name/
python -m cmd_tf -av repos/evaldata/yolov5-$run_name/
cat repos/evaldata/yolov5-$run_name/evals/evals.json