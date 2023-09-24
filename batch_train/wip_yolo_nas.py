import argparse
import ast
import json
from math import isnan
import os
import shutil
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from super_gradients import Trainer
import torch
from torch.utils.data import DataLoader

from super_gradients.training import models
from super_gradients.training import dataloaders
from super_gradients.training.datasets import YoloDarknetFormatDetectionDataset
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

from utils import *

def main():
    # Parse
    parser = argparse.ArgumentParser(prog='', description='.')
    parser.add_argument('-d','--datasets-path', type=str, default='', help='.')
    parser.add_argument('-t','--testset-path', type=str, default='', help='.')
    parser.add_argument('-o','--output-path', type=str, default='training/yolo_nas', help='.')
    parser.add_argument('-rsf','--recursive-folder-searching', action='store_true', help='.')
    
    parser.add_argument('-s','--img-size', type=int, default=640, help='Sets the img size of the model.')
    parser.add_argument('-b','--batch-size', type=int, default=8, help='Sets the batch size to train with.')
    parser.add_argument('-e','--epochs', type=int, default=100, help='Sets the epochs to train for.')
    parser.add_argument('-m','--model', type=str, default='yolo_nas_s', help='Sets the model to train with.')
    
    parser.add_argument('-wi','--worker-index', type=int, default=-1, help='.')
    parser.add_argument('-wc','--worker-count', type=int, default=-1, help='.')
    
    parser.add_argument('-db','--debug', action='store_true', help='.')
    
    args = parser.parse_args()

    # Paths
    root_dir = Path(__file__).resolve().parent
    datasets_path = Path(args.datasets_path)
    datasets_dirs = [x.parent for x in datasets_path.glob('**/dataset-def.json') 
                    if (not args.recursive_folder_searching and x.parent.parent == datasets_path or args.recursive_folder_searching)
                    and not str(x).__contains__("-valset")]
    datasets_dirs.sort(key=lambda d: d.stem)
    testset_path = Path(args.testset_path)
    
    dd_n = len(datasets_dirs)
    if args.worker_index >= 0 and args.worker_count > 0:
        datasets_dirs = datasets_dirs[int((dd_n / args.worker_count) * args.worker_index):int((dd_n / args.worker_count) * (args.worker_index+1))]
    newline_char = "\n" # Python 3.9 :/
    print(f'Running ensemble run on the following {len(datasets_dirs)} datasets:\n{newline_char.join([str(x) for x in datasets_dirs])}')
    #sys.exit(0) # For dataset choosing testing
    
    os.system(f'python traindata-creator/fixYolo5Yamls.py -df {datasets_path}')
    
    # Train
    start_time = time.time()

    loop_folders = datasets_dirs if not args.debug else datasets_dirs[:1]
    for dataset_dir in loop_folders:
        yolo_nas_train_loop(dataset_dir, 
                            testset_path, 
                            run_name=dataset_dir.stem,
                            output_path=args.output_path,
                            epochs=args.epochs,
                            img_size=args.img_size,
                            batch_size=args.batch_size,
                            model_name=args.model
                            )
        
    end_time = time.time()
    diff_time = end_time  - start_time
    parsed_time = time.strftime("%H:%M:%S", time.gmtime(diff_time))
    write_textfile(f'{diff_time}\n{parsed_time}', Path(args.output_path) / 'train_time.txt')
    print(f'Training took: {parsed_time}')

def yolo_nas_train_loop(dataset_path, 
                        valset_path, 
                        output_path = 'training/yolo_nas',
                        run_name = 'default', 
                        img_size = 640, 
                        batch_size = 8, 
                        epochs = 100, 
                        model_name = 'yolo_nas_s'
                        ):
    # Set Paths
    project_folder = Path(output_path)
    training_run_folder = project_folder / run_name
    training_run_testdata_folder = training_run_folder / 'test'
    dataset_path = Path(dataset_path)
    valset_path = Path(valset_path)
    print('Training in: ', dataset_path)
    # Gen training def json
    dataset_def_dict = json.loads(read_textfile(dataset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    valset_def_dict = json.loads(read_textfile(valset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    train_def_dict = {
        'run_name': run_name,
        'disabled_yolo_aug': False,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'model': model_name,
        'dataset': dataset_def_dict,
        'valset': valset_def_dict,
    }
    
    classes = ["marker"]
    train_dataloader = dataloaders.get(name='coco2017_train',
        dataset_params={
            "data_dir": dataset_path,
            },
        dataloader_params={'num_workers': 2}
    )
    val_dataloader = dataloaders.get(name='coco2017_val',
        dataset_params={
            "data_dir": dataset_path,
            },
        dataloader_params={'num_workers': 2}
    )
    
    #train_data.dataset.plot(plot_transformed_data=True)

    print('--- Training...')
    
    trainer = Trainer(
        experiment_name=run_name, 
        ckpt_root_dir=training_run_folder
    )
 
    model = models.get(
        model_name, 
        num_classes=len(classes), 
        pretrained_weights="coco"
    )
    
    # Taken from: https://learnopencv.com/train-yolo-nas-on-custom-dataset/
    train_params = {
        'silent_mode': False,
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": epochs,
        #"mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(classes),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(classes),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            ),
            DetectionMetrics_050_095(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(classes),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50:0.95'
    }
    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_dataloader, 
        valid_loader=val_dataloader
    )
    
    print('--- Evaluating...')
    model_out = model.predict(get_files_from_folders_with_ending([valset_path / 'val' / 'images'], '.png'))
    model_out.show()
    
    #os.system(f'python batch_train/yolov8_evaluate.py -r {training_run_folder} -t {valset_path}')
    write_textfile(json.dumps(train_def_dict, indent=4), training_run_folder / 'training-def.json')

if __name__ == '__main__':
    main()