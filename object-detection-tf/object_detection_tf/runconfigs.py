from dataclasses import dataclass
import tensorflow as tf
from tensorflow import optimizers as optis
from keras import losses as loss
from keras.callbacks import LearningRateScheduler

from object_detection_tf.loss import dice_coef_loss, dice_coef_2_loss, bcross_mod_fun_loss, BinaryCrossEntropyLoss, BinaryCrossEntropyLossFlat

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    decay_rate=0.9)

@dataclass
class RunConfig:
    name: str
    loss: any
    optimizer: any
    callbacks: list[tf.keras.callbacks.Callback]
    normalize_input_data: bool = False

# ---------------------------------------------------------------------------------------------------------------------------------------------

configs = [
    RunConfig( "dice",              dice_coef_loss,                             "adam", []),
    RunConfig( "dice-cadam",        dice_coef_loss,                             "adam", [LearningRateScheduler(lr_schedule, verbose=1)]),
    
    RunConfig( "bcross",            "binary_crossentropy",                      "adam", []),
    ]

# ---------------------------------------------------------------------------------------------------------------------------------------------

def load_runconfig(runname: str = ""):
    cur_conf: RunConfig = None
    
    configs.reverse()
    for conf in configs:
        if runname.startswith(conf.name):
            cur_conf = conf
            break
            
    if cur_conf is None:
        cur_conf = configs[-1] # First conf is default
        
    return cur_conf