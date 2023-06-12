import os
from pathlib import Path
import random
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
import segmentation_models as sm

from cmd_tf.utility import get_files_from_folders_with_ending

# --- Dataloader ---------------------------------------------------------------------------------------
BACKBONE = 'efficientnetb3'
CLASSES = ['marker']
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
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
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

def get_sm_traindata(dataset_dir, batch_size, img_size, print_data = False):
    train_data_dir = dataset_dir / 'train'
    train_x_paths = get_files_from_folders_with_ending([train_data_dir], "_in.png")
    train_y_paths = get_files_from_folders_with_ending([train_data_dir], "_seg.png")
    random.Random(1337).shuffle(train_x_paths)
    random.Random(1337).shuffle(train_y_paths)
    
    if print_data:
        print("Train in imgs:", train_x_paths.__len__(), "| Train target imgs:", train_y_paths.__len__())
        for input_path, target_path in zip(train_x_paths[:3], train_y_paths[:3]):
            print(os.path.basename(input_path), "|", os.path.basename(target_path))
    
    # Dataset for train images
    train_dataset = Dataset(
        train_x_paths, 
        train_y_paths, 
        classes=CLASSES, 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_gen = Dataloder(train_dataset, batch_size=batch_size, shuffle=True)

    # check shapes for errors
    assert train_gen[0][0].shape == (batch_size, 320, 320, 3)
    assert train_gen[0][1].shape == (batch_size, 320, 320, n_classes)
    
    return train_gen, train_x_paths, train_y_paths, train_dataset
    
def get_sm_valdata(dataset_dir, batch_size, img_size, print_data = False):
    val_data_dir = dataset_dir / 'val'
    val_x_paths = get_files_from_folders_with_ending([val_data_dir], "_in.png")
    val_y_paths = get_files_from_folders_with_ending([val_data_dir], "_seg.png")
    random.Random(420).shuffle(val_x_paths)
    random.Random(420).shuffle(val_y_paths)
    
    if print_data:
        print("Val in imgs:", val_x_paths.__len__(), "| Val target imgs:", val_y_paths.__len__())
        for input_path, target_path in zip(val_x_paths[:3], val_y_paths[:3]):
            print(os.path.basename(input_path), "|", os.path.basename(target_path))
            
    # Dataset for validation images
    val_dataset = Dataset(
        val_x_paths, 
        val_y_paths, 
        classes=CLASSES, 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    
    val_gen = Dataloder(val_dataset, batch_size=1, shuffle=False)
    
    return val_gen, val_x_paths, val_y_paths, val_dataset

# --- Loss ---------------------------------------------------------------------------------------
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
LR = 0.0001

sm_dice_x_bfocal_loss = dice_loss + (1 * focal_loss)
sm_metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
sm_optim = keras.optimizers.Adam(LR)

# --- Model ---------------------------------------------------------------------------------------
sm_unet_model = sm.Unet(BACKBONE, classes=1, activation=('sigmoid' if n_classes == 1 else 'softmax'))
sm_linknet_model = sm.Linknet(BACKBONE, classes=1, activation=('sigmoid' if n_classes == 1 else 'softmax'))
