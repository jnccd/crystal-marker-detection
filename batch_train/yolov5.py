import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

from utility import *

def main():
    # Parse
    parser = argparse.ArgumentParser(prog='', description='.')
    parser.add_argument('-d','--dataset-path', type=str, default='', help='.')
    parser.add_argument('-t','--testset-path', type=str, default='', help='.')
    parser.add_argument('-o','--output-path', type=str, default='training/yolov5', help='.')
    parser.add_argument('-rsf','--recursive-folder-searching', action='store_true', help='.')
    
    parser.add_argument('-s','--img-size', type=int, default=640, help='Sets the img size of the model.')
    parser.add_argument('-b','--batch-size', type=int, default=8, help='Sets the batch size to train with.')
    parser.add_argument('-e','--epochs', type=int, default=100, help='Sets the epochs to train for.')
    parser.add_argument('-m','--model', type=str, default='yolov5s', help='Sets the model to train with.')
    parser.add_argument('-de','--device', type=str, default='0', help='Sets the device to train on.')
    parser.add_argument('-rw','--init-random-weights', action='store_true', help='.')
    parser.add_argument('-snr','--skip-noaug-runs', action='store_true', help='.')
    
    parser.add_argument('-ct','--confidence-threshold', type=float, default=0.5, help='The minimum confidence of considered predictions.')
    parser.add_argument('-bis','--border-ignore-size', type=float, default=0, help='Ignore markers at the border of the image, given in widths from 0 to 0.5.')
    parser.add_argument('-us','--use-sahi', action='store_true', help='Use Sahi for inference.')
    
    parser.add_argument('-wi','--worker-index', type=int, default=-1, help='.')
    parser.add_argument('-wc','--worker-count', type=int, default=-1, help='.')
    
    parser.add_argument('-db','--debug', action='store_true', help='.')
    
    args = parser.parse_args()

    # Paths
    root_dir = Path(__file__).resolve().parent
    datasets_path = root_dir.parent / args.dataset_path
    datasets_dirs = [x.parent for x in datasets_path.glob('**/yolov5-*.yaml') 
                    if (not args.recursive_folder_searching and x.parent.parent == datasets_path or args.recursive_folder_searching)
                    and not str(x).__contains__("-valset")]
    datasets_dirs.sort(key=lambda d: d.stem)
    testset_path = root_dir.parent / args.testset_path
    output_folder = create_dir_if_not_exists(Path(Path(args.output_path)))
    
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
        # Without yolov5 aug
        if not args.skip_noaug_runs:
            yolov5_train_loop(
                dataset_dir, 
                testset_path, 
                run_name=dataset_dir.stem,
                output_path=args.output_path,
                epochs=args.epochs,
                img_size=args.img_size,
                batch_size=args.batch_size,
                model=args.model,
                device=args.device,
                init_random_weights=args.init_random_weights,
                no_aug=True,
                use_sahi=args.use_sahi,
                border_ignore_size=args.border_ignore_size,
                confidence_threshold=args.confidence_threshold,
            )
        # With yolov5 aug
        yolov5_train_loop(
            dataset_dir, 
            testset_path, 
            run_name=dataset_dir.stem+'-yolo5aug',
            output_path=args.output_path,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            model=args.model,
            device=args.device,
            init_random_weights=args.init_random_weights,
            no_aug=False,
            use_sahi=args.use_sahi,
            border_ignore_size=args.border_ignore_size,
            confidence_threshold=args.confidence_threshold,
        )
        
    end_time = time.time()
    diff_time = end_time  - start_time
    parsed_time = time.strftime("%H:%M:%S", time.gmtime(diff_time))
    write_textfile(f'{diff_time}\n{parsed_time}', output_folder / 'train_time.txt')
    print(f'Training took: {parsed_time}')

def yolov5_train_loop(dataset_path, 
                      valset_path, 
                      output_path = 'training/yolov5',
                      run_name = 'default', 
                      img_size = 640, 
                      batch_size = 8, 
                      epochs = 100, 
                      device = '0',
                      model = 'yolov5s', 
                      init_random_weights = False, 
                      no_aug = False,
                      use_sahi = False,
                      border_ignore_size = 0,
                      confidence_threshold = 0.5,
                      ):
    # --- Set Paths
    project_folder = Path(output_path)
    training_run_folder = project_folder / run_name
    training_run_testdata_folder = training_run_folder / 'test'
    dataset_path = Path(dataset_path)
    valset_path = Path(valset_path)
    # --- Gen training def json
    dataset_def_dict = json.loads(read_textfile(dataset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    valset_def_dict = json.loads(read_textfile(valset_path / 'dataset-def.json').replace("    ", "").replace("\n", ""))
    train_def_dict = {
        'run_name': run_name,
        'disabled_yolo_aug': no_aug,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'model': model,
        'dataset': dataset_def_dict,
        'valset': valset_def_dict,
    }

    # --- Set extra arguments
    yolov5_args = ''

    # Random yolov5 weight init
    if init_random_weights:
        yolov5_args += f"--weights '' --cfg {model}.yaml "
    else:
        yolov5_args += f'--weights {model}.pt '

    # Disable yolov5 augmentation
    if no_aug:
        yolov5_args += '--hyp hyp.no-augmentation.yaml '
        
    # Set device
    yolov5_args += f'--device {device} '
    
    # Overwriting other training is okay
    yolov5_args += f'--exist-ok '
    
    # --- Commands
    print('--- Training...')
    os.system(f'python repos/yolov5/train.py --name {run_name} --img {img_size} --batch {batch_size} --epochs {epochs} --project {project_folder} --data {dataset_path}/{dataset_path.stem}.yaml {yolov5_args}')
    os.system(f'rm {model}.pt')
    print('--- Evaluating...')
    os.system(f'python batch_train/yolov5_evaluate.py -r {training_run_folder} -t {valset_path}/ {"-us" if use_sahi else ""} -bis {border_ignore_size} -ct {confidence_threshold}')
    write_textfile(json.dumps(train_def_dict, indent=4), training_run_folder / 'training-def.json')

if __name__ == '__main__':
    main()