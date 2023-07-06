import ast
import json
import os
import math
import random
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img, array_to_img

from cmd_tf.runconfigs import load_runconfig

num_classes = 1

def fit(
    batch_size: int = 16, 
    num_epochs: int = 1, 
    run: str = 'default', 
    data_folder: str = 'renders', 
    size: int = 160, 
    print_model: bool = False, 
    use_multi_gpu_strategy: bool = False, 
    extra_settings = {},
    ):

    img_size = (size, size)
    
    # --- Paths ---
    # Base
    root_dir = Path(__file__).resolve().parent
    dataset_dir = Path(data_folder)
    # MIP Server Base Paths
    if not os.path.isdir(dataset_dir):
        root_dir = Path("/data/cmdtf")
        dataset_dir = root_dir / data_folder
    # Set Run Paths
    runs_dir = root_dir / 'runs'
    run_dir = runs_dir / f'{run}'
    val_dir = run_dir / 'validation'
    weights_dir = run_dir / 'weights'
    
    # Load runconfig
    cur_conf = load_runconfig(run)
    (get_traindata, get_valdata) = cur_conf.dataset_loader
    
    # Prepare Training and Validation Data
    train_gen, train_x_paths, _, _ = get_traindata(dataset_dir, batch_size, img_size, extra_settings)
    val_gen, _, _, _ = get_valdata(dataset_dir, batch_size, img_size, extra_settings)
    epoch_steps = math.floor(len(train_x_paths) / batch_size)
    
    print()
    print("---Configuration---------------------------------")
    print("Batch Size:", batch_size)
    print("Epochs:", num_epochs)
    print("Run Name:", run)
    print("Run-Config:", cur_conf.name)
    print("Loss:", cur_conf.loss)
    print("Optimizer:", cur_conf.optimizer)
    print("Steps per Epoch:", epoch_steps)
    print("Extra settings:", extra_settings)
    print("--------------------------------------------------")
    print()

    print("Build model...")
    keras.backend.clear_session() # Free up RAM in case the model definition cells were run multiple times

    if not use_multi_gpu_strategy:
        strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    with strategy.scope():
        
        # Build model
        model = cur_conf.get_model(img_size, num_classes, extra_settings)
        if print_model:
            model.summary()
            tf.keras.utils.plot_model(model, to_file=run_dir / "model.png", show_shapes=True)
        
        print("Compile model...")
        model.compile(optimizer=cur_conf.optimizer, 
                    loss=cur_conf.loss,
                    metrics=cur_conf.metrics)
        
        cur_conf.callbacks.append(
            keras.callbacks.ModelCheckpoint(weights_dir / "weights.h5", save_best_only=True)
        )
        callbacks = cur_conf.callbacks

        print("Load weights and train...")
        if os.path.isfile(weights_dir / 'weights.index'):
            print("Found preexisting weights")
            model.load_weights(weights_dir / 'weights')
        else:
            print("Learning from scratch")

        # Train
        train_start_time = timer()
        model_out = model.fit(train_gen, steps_per_epoch=epoch_steps, epochs=num_epochs, validation_data=val_gen, callbacks=callbacks, verbose=1)
        train_end_time = timer()
        print('Training took', (timedelta(seconds = train_end_time - train_start_time)))

        model.save_weights(weights_dir / 'weights')

        print("Write evaluation...")
        val_gen, _, _, _ = get_valdata(dataset_dir, batch_size, img_size, extra_settings)
        eval_results = model.evaluate(val_gen)
        # Parse and combine histories if old history exists
        history_file = run_dir / 'history'
        if os.path.exists(history_file) and os.path.isfile(history_file):
            with open(history_file) as f:
                history_str = f.read()
            full_history = ast.literal_eval(history_str)
            for key, value in full_history.items():
                full_history[key].extend(model_out.history[key])
        else:
            full_history = model_out.history
        # Write training history textfile
        with open(history_file, "w") as f:
            f.write(str(full_history))
        # Write eval textfile
        eval_file = run_dir / 'evals'
        with open(eval_file, "w") as f:
            all_metrics = [str(x) for x in cur_conf.metrics]
            all_metrics.insert(0, "Loss")
            eval_dict = {}
            for metric, res in zip(all_metrics, eval_results):
                eval_dict[metric] = res
            f.write(json.dumps(eval_dict, indent=4))
        # Write eval plots
        xc = range(1, len(list(full_history.values())[0])+1)
        for metric in full_history:
            if metric == "lr":
                train = full_history[metric]
                plt.clf()
                plt.title(f"Learn rate over epochs")
                plt.xlabel("Epochs")
                plt.ylabel(metric)
                plt.plot(xc, train)
                if num_epochs < 20:
                    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
                plt.savefig(str(eval_file) + f'_{metric}_plot.pdf', dpi=100)
            else:
                if not metric.startswith("val_"):
                    train = full_history[metric]
                    val = full_history["val_"+metric]
                    plt.clf()
                    plt.title(f"Train and Validation {metric} over epochs")
                    plt.xlabel("Epochs")
                    plt.ylabel(metric)
                    plt.plot(xc, train, label="train "+metric)
                    plt.plot(xc, val, label="validation "+metric)
                    if num_epochs < 20:
                        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
                    plt.legend(loc="upper left")
                    plt.savefig(str(eval_file) + f'_{metric}_plot.pdf', dpi=100)
        
        print("Write validation...")
        # Generate predictions for all images in the validation set
        val_gen, val_x_paths, val_y_paths, aug_data = get_valdata(dataset_dir, batch_size, img_size, extra_settings)
        val_preds = model.predict(val_gen)
        
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        # Display some results for validation images
        for i in range(0, min(len(val_preds), len(val_x_paths))):
            # Display input image
            if aug_data is None:# or True:
                in_img = ImageOps.autocontrast(load_img(val_x_paths[i]))
            else:
                in_img = ImageOps.autocontrast(array_to_img(aug_data[i][0]))
            in_img.save(val_dir / f'{i}_input.png')

            # Display ground-truth target mask
            if aug_data is None:# or True:
                gt_img = ImageOps.autocontrast(load_img(val_y_paths[i]))
            else:
                gt_img = ImageOps.autocontrast(array_to_img(aug_data[i][1]))
            gt_img.save(val_dir / f'{i}_target_output.png')

            # Display mask predicted by our model
            out_img = ImageOps.autocontrast(array_to_img(val_preds[i]))
            out_img.save(val_dir / f'{i}_network_output.png')
        
        print("Increment epoch counter...")
        # Increment run epoch counter file
        epoch_counter_file = run_dir / 'epochs'
        if not os.path.exists(epoch_counter_file):
            with open(epoch_counter_file, "w") as f:
                f.write(str(num_epochs))
        else:
            with open(epoch_counter_file, "r") as f:
                run_epochs = int(f.read())
            with open(epoch_counter_file, "w") as f:
                f.write(str(run_epochs + num_epochs))