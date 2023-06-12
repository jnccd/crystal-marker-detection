import os
import math
import random
from pathlib import Path

import numpy as np
from PIL import ImageOps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img, array_to_img

from cmd_tf.runconfigs import load_runconfig
from cmd_tf.utility import get_files_from_folders_with_ending

num_classes = 1

def fit(
    batch_size: int = 16, 
    num_epochs: int = 1, 
    run: str = 'default', 
    data_folder: str = 'renders', 
    size: int = 160, 
    print_model: bool = False, 
    use_multi_gpu_strategy: bool = False, 
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
    run_dir = runs_dir / f'run-{run}'
    val_dir = run_dir / 'validation'
    weights_dir = run_dir / 'weights'
    
    # Load runconfig
    cur_conf = load_runconfig(run)
    (get_traindata, get_valdata) = cur_conf.dataset_loader
    
    # Prepare Training and Validation Data
    train_gen, train_x_paths, _ = get_traindata(dataset_dir, batch_size, img_size)
    val_gen, _, _ = get_valdata(dataset_dir, batch_size, img_size)
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
        model = cur_conf.model
        if print_model:
            model.summary()
            tf.keras.utils.plot_model(model, to_file=run_dir / "model.png", show_shapes=True)
        
        print("Compile model...")
        metrics = [tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.Precision(),
                    ]
        model.compile(optimizer=cur_conf.optimizer, 
                    loss=cur_conf.loss,
                    metrics=metrics)
        
        # Set callbacks
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

        # Train the model, doing validation at the end of each epoch.
        model_out = model.fit(train_gen, steps_per_epoch=epoch_steps, epochs=num_epochs, validation_data=val_gen, callbacks=callbacks, verbose=1)

        model.save_weights(weights_dir / 'weights')

        print("Write evaluation...")
        # First eval textfile
        val_gen, _, _ = get_valdata(dataset_dir, batch_size, img_size)
        eval_results = model.evaluate(val_gen)
        eval_file = run_dir / 'evals'
        i = 0
        while os.path.exists(eval_file):
            eval_file = run_dir / ('evals'+str(i))
            i+=1
        with open(eval_file, "w") as f:
            f.write(str(model_out.history) + "\n\n")
            f.write(str(metrics) + "\n")
            f.write(str(eval_results) + "\n")
        # Then eval plots
        xc = range(1, num_epochs+1)
        for metric in model_out.history:
            if metric == "lr":
                train = model_out.history[metric]
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
                    train = model_out.history[metric]
                    val = model_out.history["val_"+metric]
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
        val_gen, val_x_paths, val_y_paths = get_valdata(dataset_dir, batch_size, img_size)
        val_preds = model.predict(val_gen)
        
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        # Display some results for validation images
        for i in range(0, min(50, len(val_x_paths), batch_size)):
            # Display input image
            inimg = ImageOps.autocontrast(load_img(val_x_paths[i]))
            inimg.save(val_dir / f'{i}_input.png')

            # Display ground-truth target mask
            img = ImageOps.autocontrast(load_img(val_y_paths[i]))
            img.save(val_dir / f'{i}_target_output.png')

            # Display mask predicted by our model
            img = ImageOps.autocontrast(array_to_img(val_preds[i]))
            img.save(val_dir / f'{i}_network_output.png')
        
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