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
from tensorflow.keras import backend as K

import segmentation_models as sm

NUM_CLASSES = 1
BATCH_SIZE = 8
NUM_EPOCHS = 5
IMG_SIZE = (320, 320)

def get_files_from_folder_with_ending(folders, ending):
    paths = []
    for folder in folders:
        paths.extend(sorted(
            [
                os.path.join(folder, fname)
                for fname in os.listdir(folder)
                if fname.endswith(ending)
            ]
        ))
    return paths

class ClassicBatchgen(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_paths = target_paths

    def __len__(self):
        return len(self.target_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_paths = self.target_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = load_img(path, target_size=self.img_size)
            
        y = np.zeros((self.batch_size,) + self.img_size + (NUM_CLASSES,), dtype="float32")
        for j, path in enumerate(batch_target_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, axis=2)
        return x, y
    
# Data loading
root_dir = Path(__file__).resolve().parent
val_dir = root_dir / 'xception_val'
dataseries_t1_dir = root_dir/'..'/'traindata-creator'/'dataseries-320-webcam-images-1312ecab-04e7-4f45-a714-07365d8c0dae'/'images_traindata'
dataseries_v1_dir = root_dir/'..'/'traindata-creator'/'dataseries-320-webcam-images-203d3683-7c91-4429-93b6-be24a28f47bf'/'images_traindata'
dataseries_t2_dir = root_dir/'..'/'traindata-creator'/'dataseries-320-webcam-images-f50ec0b7-f960-400d-91f0-c42a6d44e3d0'/'images_traindata'

data_img_paths = get_files_from_folder_with_ending([dataseries_t1_dir, dataseries_t2_dir], "_in.png")
data_target_paths = get_files_from_folder_with_ending([dataseries_t1_dir, dataseries_t2_dir], "_seg.png")
val_img_paths = get_files_from_folder_with_ending([dataseries_v1_dir], "_in.png")
val_target_paths = get_files_from_folder_with_ending([dataseries_v1_dir], "_seg.png")

train_gen = ClassicBatchgen(BATCH_SIZE, IMG_SIZE, data_img_paths, data_target_paths)
val_gen = ClassicBatchgen(BATCH_SIZE, IMG_SIZE, val_img_paths, val_target_paths)

# Loss
def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
def dice_coef_loss(y_true, y_pred, smooth=100):
    y_true_f = tf.cast(y_true, tf.float32)
    return 1 - dice_coef(y_true_f, y_pred, smooth)
def dice_coef2(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return ((2. * intersection + smooth) / (union + smooth)) / 2
def dice_coef2_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32) # dice_coef() requires uniform typings
    return 1 - dice_coef(y_true, y_pred)

# Data loading
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

train_prep = preprocess_input(train_gen)
val_prep = preprocess_input(val_gen)

assert train_gen[0][0].shape == (BATCH_SIZE, 320, 320, 3)
assert train_gen[0][1].shape == (BATCH_SIZE, 320, 320, NUM_CLASSES)

# Model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
print("Compile model...")
model.compile(
    'Adam',
    loss=dice_coef2_loss,
    metrics=[sm.metrics.iou_score],
)

epoch_steps = math.floor((len(data_img_paths) - len(val_img_paths)) / BATCH_SIZE)
model_out = model.fit(train_prep, validation_data=val_prep, steps_per_epoch=epoch_steps, epochs=NUM_EPOCHS, callbacks=[], verbose=1)

model.save_weights('./weights')

val_gen = ClassicBatchgen(BATCH_SIZE, IMG_SIZE, val_img_paths, val_target_paths)
val_preds = model.predict(val_gen)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Display some results for validation images
for i in range(0, min(50, len(val_img_paths))):
    # Display input image
    inimg = ImageOps.autocontrast(load_img(val_img_paths[i]))
    inimg.save(val_dir / f'{i}_input.png')

    # Display ground-truth target mask
    img = ImageOps.autocontrast(load_img(val_target_paths[i]))
    img.save(val_dir / f'{i}_target_output.png')

    # Display mask predicted by our model
    img = ImageOps.autocontrast(array_to_img(val_preds[i]))
    img.save(val_dir / f'{i}_network_output.png')

#image = tf.image.decode_png(tf.io.read_file('file.png'))
#result = model(image)