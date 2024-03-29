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
import torch
from torch.utils.data import DataLoader

from utility import *

def main():
    global Trainer, models, dataloaders, DetectionMetrics_050, DetectionMetrics_050_095, PPYoloELoss, PPYoloEPostPredictionCallback
    
    # Parse
    parser = argparse.ArgumentParser(prog='yolo-nas-batch-train', description='Train and test yolo nas models on multiple datasets at once.')
    parser.add_argument('-d','--datasets-path', type=str, default='', help='The path to the folder containing coco dataset folders to train from.')
    parser.add_argument('-t','--testset-path', type=str, default='', help='The path to the coco testset to use the validation data of for testing.')
    parser.add_argument('-o','--output-path', type=str, default='training/yolo_nas', help='The folder in which the training run folder will be placed.')
    parser.add_argument('-rsf','--recursive-folder-searching', action='store_true', help='Doesnt check for folder depth of found dataset folders from the datasets-path.')
    
    parser.add_argument('-s','--img-size', type=int, default=640, help='Sets the img size of the model.')
    parser.add_argument('-b','--batch-size', type=int, default=8, help='Sets the batch size to train with.')
    parser.add_argument('-e','--epochs', type=int, default=100, help='Sets the epochs to train for.')
    parser.add_argument('-m','--model', type=str, default='yolo_nas_s', help='Sets the model to train with.')
    
    parser.add_argument('-wi','--worker-index', type=int, default=-1, help='For multi gpu server runs, this sets which datasets of all found ones should be worked on by this instance of batch train.')
    parser.add_argument('-wc','--worker-count', type=int, default=-1, help='For multi gpu server runs, this sets which datasets of all found ones should be worked on by this instance of batch train.')
    
    parser.add_argument('-db','--debug', action='store_true', help='Generate more output.')
    
    args = parser.parse_args()

    # Paths
    root_dir = Path(__file__).resolve().parent
    datasets_path = Path(args.datasets_path)
    datasets_dirs = [x.parent for x in datasets_path.glob('**/dataset-def.json') 
                    if (not args.recursive_folder_searching and x.parent.parent == datasets_path or args.recursive_folder_searching)
                    and not str(x).__contains__("-valset")]
    datasets_dirs.sort(key=lambda d: d.stem) # Necessary on Linux Python because glob doesn't return sorted folders there
    testset_path = Path(args.testset_path)
    
    # Filter dataset dirs that this instance is not responsible for
    dd_n = len(datasets_dirs)
    if args.worker_index >= 0 and args.worker_count > 0:
        datasets_dirs = datasets_dirs[int((dd_n / args.worker_count) * args.worker_index):int((dd_n / args.worker_count) * (args.worker_index+1))]
    newline_char = "\n" # Python 3.9 :/
    print(f'Running ensemble run on the following {len(datasets_dirs)} datasets:\n{newline_char.join([str(x) for x in datasets_dirs])}')
    #sys.exit(0) # For dataset choosing testing
    
    # Redirect Torch hub cache support on
    os.system('mkdir ./.cache')
    os.environ['TORCH_HOME'] = './.cache'
    
    # Set log position env var and get super_gradients imports
    os.environ["SUPER_GRADIENTS_LOG_DIR"] = str(Path(args.output_path) / f'sg_logs_{time.time()}')
    from super_gradients import Trainer
    from super_gradients.training import models 
    from super_gradients.training import dataloaders
    from super_gradients.training.metrics import (
        DetectionMetrics_050,
        DetectionMetrics_050_095
    )
    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
    
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
    global Trainer, models, dataloaders, DetectionMetrics_050, DetectionMetrics_050_095, PPYoloELoss, PPYoloEPostPredictionCallback
    
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
    
    # clear training run folder
    esc_char = '\\'
    os.system(f'rm -r {str(training_run_folder).replace(esc_char, "/")}/RUN_*')
    
    classes = ["marker"]
    train_dataloader = dataloaders.get(name='coco2017_train',
        dataset_params={
            "data_dir": dataset_path,
            },
        dataloader_params={
            'batch_size':batch_size,
            'num_workers': 2,
            }
    )
    val_dataloader = dataloaders.get(name='coco2017_val',
        dataset_params={
            "data_dir": dataset_path,
            },
        dataloader_params={
            'batch_size':batch_size,
            'num_workers': 2,
            }
    )
    
    train_dataloader.dataset.transforms.pop(2)
    #train_dataloader.dataset.plot(plot_transformed_data=True)

    print('--- Training...')
    
    trainer = Trainer(
        experiment_name=run_name, 
        ckpt_root_dir=project_folder,
    )
 
    model = models.get(
        model_name, 
        num_classes=len(classes), 
        pretrained_weights="coco"
    )
    
    # Based on: https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/roboflow_yolo_nas_s.yaml
    train_params = {
        'silent_mode': False,
        #"average_best_models":True,
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
        "mixed_precision": True,
        "loss": PPYoloELoss(
            #use_static_assigner=False,
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
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }
    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_dataloader, 
        valid_loader=val_dataloader
    )
    
    print('--- Evaluating...')
    os.system(f'python batch_train/yolo_nas_evaluate.py -r {training_run_folder} -t {valset_path}')
    write_textfile(json.dumps(train_def_dict, indent=4), training_run_folder / 'training-def.json')

if __name__ == '__main__':
    main()