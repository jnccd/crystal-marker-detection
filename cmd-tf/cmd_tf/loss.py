import keras.backend as K
import tensorflow as tf
import numpy as np

def flat_dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return ((2. * intersection + smooth) / (union + smooth)) / 2 # Divide by two because it normally gives outputs from 0 to 2 for whatever reason but its the only one that works
def flat_dice_coef_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32) # dice_coef() requires uniform typings
    return 1 - flat_dice_coef(y_true, y_pred)