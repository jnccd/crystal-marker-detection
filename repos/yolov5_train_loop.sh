#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

run_name=default
dataset_name=yolov5-640-aug-good-pics-v2
valset_name=yolov5-640-on-skin-valset
project_folder=repos/training/yolov5

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
if [[ $run_name == *nhyp ]]; then
    hyp='--hyp hyp.no-augmentation.yaml'
else
    hyp=''
fi
echo "hyp is '$hyp'!"

python repos/yolov5/train.py --name $run_name --img 640 --batch 8 --epochs 300 --project $project_folder --data traindata-creator/dataset/$dataset_name/$dataset_name.yaml --weights yolov5s.pt $hyp
rm ./yolov5s.pt
python repos/yolov5_gen_evaldata.py -r $run_name -df traindata-creator/dataset/$valset_name/
python -m cmd_tf -av repos/evaldata/yolov5/$run_name/
cat repos/evaldata/yolov5/$run_name/evals/evals.json