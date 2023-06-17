import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img
import keras.backend as K

from cmd_tf.utility import get_files_from_folders_with_ending

# --- Dataloader ---------------------------------------------------------------------------------------
class XUnetBatchgen(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            y[j] = load_img(path, target_size=self.img_size, color_mode="grayscale")
        return x, y
    
def get_xunet_traindata(dataset_dir, batch_size, img_size, extra_settings, print_data = False):
    train_data_dir = dataset_dir / 'train'
    train_x_paths = get_files_from_folders_with_ending([train_data_dir], "_in.png")
    train_y_paths = get_files_from_folders_with_ending([train_data_dir], "_seg.png")
    random.Random(1337).shuffle(train_x_paths)
    random.Random(1337).shuffle(train_y_paths)
    
    if print_data:
        print("Train in imgs:", train_x_paths.__len__(), "| Train target imgs:", train_y_paths.__len__())
        for input_path, target_path in zip(train_x_paths[:3], train_y_paths[:3]):
            print(os.path.basename(input_path), "|", os.path.basename(target_path))
    
    train_gen = XUnetBatchgen(batch_size, img_size, train_x_paths, train_y_paths)
    
    return train_gen, train_x_paths, train_y_paths, None
    
def get_xunet_valdata(dataset_dir, batch_size, img_size, extra_settings, print_data = False):
    val_data_dir = dataset_dir / 'val'
    val_x_paths = get_files_from_folders_with_ending([val_data_dir], "_in.png")
    val_y_paths = get_files_from_folders_with_ending([val_data_dir], "_seg.png")
    random.Random(420).shuffle(val_x_paths)
    random.Random(420).shuffle(val_y_paths)
    
    if print_data:
        print("Val in imgs:", val_x_paths.__len__(), "| Val target imgs:", val_y_paths.__len__())
        for input_path, target_path in zip(val_x_paths[:3], val_y_paths[:3]):
            print(os.path.basename(input_path), "|", os.path.basename(target_path))
    
    val_gen = XUnetBatchgen(batch_size, img_size, val_x_paths, val_y_paths)
    
    return val_gen, val_x_paths, val_y_paths, None
    
# --- Loss ---------------------------------------------------------------------------------------
def flat_dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return ((2. * intersection + smooth) / (union + smooth)) / 2 # Divide by two because it normally gives outputs from 0 to 2 for whatever reason but its the only one that works
def flat_dice_coef_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32) # dice_coef() requires uniform typings
    return 1 - flat_dice_coef(y_true, y_pred)

# --- Model ---------------------------------------------------------------------------------------
def get_xunet_model(img_size, num_classes, extra_settings):
    inputs = keras.Input(shape=img_size + (3,))
    
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model