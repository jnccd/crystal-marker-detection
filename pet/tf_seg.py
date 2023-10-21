from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
from timeit import default_timer as timer
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import ImageOps, Image

import albumentations as A
import segmentation_models as sm
import tensorflow as tf 
from tensorflow.keras.utils import load_img, array_to_img

from utils import *

EPOCHS = 400
BATCH_SIZE = 64
DEVICE = "cuda"
DIM_KEYPOINTS = 2
NUM_KEYPOINTS = 4
IMG_SIZE = 160
CLASSES = ['marker']
MODEL = 'unet'

print_model = False

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/segpet-0-man-pet-v2'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'
output_folder = create_dir_if_not_exists(root_dir / 'output/tf-seg')
eval_folder = create_dir_if_not_exists(output_folder / 'eval')

# --- Dataloader ----------------------------------------------------------------------------------------
BACKBONE = 'efficientnetb3'
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)

class Dataset:
    """Marker Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_fps, 
            masks_fps, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = images_fps
        self.masks_fps = masks_fps
        
        # convert str names to class values on masks
        self.class_values = [255]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            print('not binary')
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
            
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(img_size_wh):
    train_transform = [
        A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
        A.RandomRotate90(),
        # A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.05, rotate_limit=270, scale_limit=0.01, p=1),
        A.Perspective(scale=(0.01, 0.06)),
        # A.Affine(shear=(-20, 20))
        # A.HueSaturationValue(),
        # A.ColorJitter(),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(img_size_wh):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(min_height=img_size_wh[1], min_width=img_size_wh[0])
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)
            
preprocess_input = sm.get_preprocessing(BACKBONE)

def get_sm_traindata(dataset_dir, batch_size, img_size_wh, print_data = False):
    train_data_dir = dataset_dir / 'train'
    train_x_paths = get_files_from_folders_with_ending([train_data_dir], "_in.png")
    train_y_paths = get_files_from_folders_with_ending([train_data_dir], "_seg.png")
    random.Random(1337).shuffle(train_x_paths)
    random.Random(1337).shuffle(train_y_paths)
    
    if print_data:
        print("Train in imgs:", train_x_paths.__len__(), "| Train target imgs:", train_y_paths.__len__())
        for input_path, target_path in zip(train_x_paths[:3], train_y_paths[:3]):
            print(os.path.basename(input_path), "|", os.path.basename(target_path))
            
    data_aug = get_training_augmentation(img_size_wh)
    
    # Dataset for train images
    train_dataset = Dataset(
        train_x_paths, 
        train_y_paths, 
        classes=CLASSES, 
        augmentation=data_aug,
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_gen = Dataloder(train_dataset, batch_size=batch_size, shuffle=True)

    # check shapes for errors
    #print(train_gen[0][0].shape, img_size_wh)
    assert train_gen[0][0].shape == (batch_size, img_size_wh[0], img_size_wh[1], 3)
    assert train_gen[0][1].shape == (batch_size, img_size_wh[0], img_size_wh[1], n_classes)
    
    return train_gen, train_x_paths, train_y_paths, train_dataset
    
def get_sm_valdata(dataset_dir, batch_size, img_size_wh, print_data = False):
    val_data_dir = dataset_dir / 'val'
    val_x_paths = get_files_from_folders_with_ending([val_data_dir], "_in.png")
    val_y_paths = get_files_from_folders_with_ending([val_data_dir], "_seg.png")
    random.Random(420).shuffle(val_x_paths)
    random.Random(420).shuffle(val_y_paths)
    
    if print_data:
        print("Val in imgs:", val_x_paths.__len__(), "| Val target imgs:", val_y_paths.__len__())
        for input_path, target_path in zip(val_x_paths[:3], val_y_paths[:3]):
            print(os.path.basename(input_path), "|", os.path.basename(target_path))
    
    data_aug = get_validation_augmentation(img_size_wh)
    
    # Dataset for validation images
    val_dataset = Dataset(
        val_x_paths, 
        val_y_paths, 
        classes=CLASSES, 
        augmentation=data_aug,
        preprocessing=get_preprocessing(preprocess_input),
    )
    
    val_gen = Dataloder(val_dataset, batch_size=1, shuffle=False)
    
    return val_gen, val_x_paths, val_y_paths, val_dataset
    
# --- Model ----------------------------------------------------------------------------------------

def get_model():
    if MODEL == 'unet':
        return sm.Unet('resnet34', classes=n_classes, activation=('sigmoid' if n_classes == 1 else 'softmax'))
        
    return model

# --- Loss ----------------------------------------------------------------------------------------

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
LR = 0.0001

sm_dice_x_bfocal_loss = dice_loss + (1 * focal_loss)
sm_metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
sm_optim = keras.optimizers.Adam(LR)

# --- Train ----------------------------------------------------------------------------------------

# Prepare Training and Validation Data
train_gen, train_x_paths, _, _ = get_sm_traindata(dataset_dir, BATCH_SIZE, (IMG_SIZE, IMG_SIZE))
val_gen, _, _, _ = get_sm_valdata(dataset_dir, BATCH_SIZE, (IMG_SIZE, IMG_SIZE))
epoch_steps = math.floor(len(train_x_paths) / BATCH_SIZE)

print("Build model...")
keras.backend.clear_session() # Free up RAM in case the model definition cells were run multiple times

strategy = tf.distribute.get_strategy()
#strategy = tf.distribute.experimental.CentralStorageStrategy()
with strategy.scope():
    
    # Build model
    model = get_model()
    if print_model:
        model.summary()
        tf.keras.utils.plot_model(model, to_file=output_folder / "model.png", show_shapes=True)
    
    print("Compile model...")
    model.compile(optimizer=sm_optim, 
                loss=sm_dice_x_bfocal_loss,
                metrics=sm_metrics)
    
    callbacks = [keras.callbacks.ReduceLROnPlateau(),
                 keras.callbacks.ModelCheckpoint(output_folder / "weights.h5", save_best_only=True)]

    print("Load weights and train...")
    if os.path.isfile(output_folder / 'weights.index'):
        print("Found preexisting weights")
        model.load_weights(output_folder / 'weights')
    else:
        print("Learning from scratch")

    # Train
    train_start_time = timer()
    model_out = model.fit(train_gen, steps_per_epoch=epoch_steps, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1)
    train_end_time = timer()
    print('Training took', (timedelta(seconds = train_end_time - train_start_time)))

    model.save_weights(output_folder / 'weights')

    print("Write evaluation...")
    val_gen, _, _, _ = get_sm_valdata(dataset_dir, BATCH_SIZE, (IMG_SIZE, IMG_SIZE))
    eval_results = model.evaluate(val_gen)
    # Write training history textfile
    history_file = output_folder / 'history'
    full_history = model_out.history
    with open(history_file, "w") as f:
        f.write(str(full_history))
    # Write eval textfile
    eval_file = output_folder / 'evals'
    with open(eval_file, "w") as f:
        all_metrics = [str(x) for x in sm_metrics]
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
            if EPOCHS < 20:
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
                if EPOCHS < 20:
                    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
                plt.legend(loc="upper left")
                plt.savefig(str(eval_file) + f'_{metric}_plot.pdf', dpi=100)
    
    print("Write validation...")
    # Generate predictions for all images in the validation set
    val_gen, val_x_paths, val_y_paths, aug_data = get_sm_valdata(dataset_dir, BATCH_SIZE, (IMG_SIZE, IMG_SIZE))
    val_preds = model.predict(val_gen)
    
    # Display some results for validation images
    for i in range(0, min(len(val_preds), len(val_x_paths))):
        # Display input image
        if aug_data is None:# or True:
            in_img = ImageOps.autocontrast(load_img(val_x_paths[i]))
        else:
            in_img = ImageOps.autocontrast(array_to_img(aug_data[i][0]))
        in_img.save(eval_folder / f'{i}_input.png')

        # Display ground-truth target mask
        if aug_data is None:# or True:
            gt_img = ImageOps.autocontrast(load_img(val_y_paths[i]))
        else:
            gt_img = ImageOps.autocontrast(array_to_img(aug_data[i][1]))
        gt_img.save(eval_folder / f'{i}_target_output.png')

        # Display mask predicted by our model
        out_img = ImageOps.autocontrast(array_to_img(val_preds[i]))
        out_img.save(eval_folder / f'{i}_network_output.png')

