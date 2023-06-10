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

from cmd_tf.runconfigs import RunConfig, configs, load_runconfig
from cmd_tf.dataloader import ResnetBatchgen

num_classes = 1

def fit(
    batch_size: int = 32, 
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
    render_dir = root_dir / data_folder
    # Server Base Paths
    if not os.path.isdir(render_dir):
        root_dir = Path("/data/cmdtf")
        render_dir = root_dir / data_folder
    # Training Data Paths
    input_dir = render_dir / 'images'
    target_dir = render_dir / 'segmentations'
    # Set Run Paths
    runs_dir = root_dir / 'runs'
    run_dir = runs_dir / f'run-{run}'
    val_dir = run_dir / 'validation'
    weights_dir = run_dir / 'weights'
    # Prepare Training Data Img Paths
    num_samples = os.listdir(input_dir).__len__()
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png")
        ]
    )
    
    # Load runconfig
    cur_conf = load_runconfig(run)
    
    print()
    print("---Configuration---------------------------------")
    print("Number of samples:", num_samples)
    print("Batch Size:", batch_size)
    print("Epochs:", num_epochs)
    print("Run Name:", run)
    print("Run-Config:", cur_conf.name)
    print("Loss:", cur_conf.loss)
    print("Optimizer:", cur_conf.optimizer)
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
            

        # Split our img paths into a training and a validation set
        print("Create img sets...")
        train_input_img_paths = [ img for img in input_img_paths if not startswith_any(os.path.basename(img), val_prefixes) ]
        train_target_img_paths = [ img for img in target_img_paths if not startswith_any(os.path.basename(img), val_prefixes) ]
        val_input_img_paths = [ img for img in input_img_paths if startswith_any(os.path.basename(img), val_prefixes) ]
        val_target_img_paths = [ img for img in target_img_paths if startswith_any(os.path.basename(img), val_prefixes) ]
        
        random.Random(1337).shuffle(train_input_img_paths)
        random.Random(1337).shuffle(train_target_img_paths)
        random.Random(420).shuffle(val_input_img_paths)
        random.Random(420).shuffle(val_target_img_paths)
        
        print()
        print("---Train/Val Data Validation---------------------------------")
        
        print("Train in imgs:", train_input_img_paths.__len__(), "| Train target imgs:", train_target_img_paths.__len__())
        for input_path, target_path in zip(train_input_img_paths[:3], train_target_img_paths[:3]):
            print(os.path.basename(input_path), "|", os.path.basename(target_path))
        
        print("Val in imgs:", val_input_img_paths.__len__(), "| Val target imgs:", val_target_img_paths.__len__())
        for input_path, target_path in zip(val_input_img_paths[:3], val_target_img_paths[:3]):
            print(os.path.basename(input_path), "|", os.path.basename(target_path))
        
        val_samples = val_input_img_paths.__len__()
        epoch_steps = math.floor((num_samples - val_samples) / batch_size)
        print("Steps per Epoch:", epoch_steps)
        
        print("-------------------------------------------------------------")
        print()

        # Instantiate data Sequences for each split
        train_gen = ResnetBatchgen(batch_size, img_size, train_input_img_paths, train_target_img_paths, cur_conf)
        val_gen = ResnetBatchgen(batch_size, img_size, val_input_img_paths, val_target_img_paths, cur_conf)


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
        val_gen = ResnetBatchgen(batch_size, img_size, val_input_img_paths, val_target_img_paths, cur_conf)
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
        val_gen = ResnetBatchgen(batch_size, img_size, val_input_img_paths, val_target_img_paths, cur_conf)
        val_preds = model.predict(val_gen)
        
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        # Display some results for validation images
        for i in range(0, min(50, val_samples)):
            # Display input image
            inimg = ImageOps.autocontrast(load_img(val_input_img_paths[i]))
            inimg.save(val_dir / f'{i}_input.png')

            # Display ground-truth target mask
            img = ImageOps.autocontrast(load_img(val_target_img_paths[i]))
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