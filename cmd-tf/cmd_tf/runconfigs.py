from dataclasses import dataclass
import tensorflow as tf
from tensorflow import optimizers as optis
from keras import losses as loss
from keras.callbacks import LearningRateScheduler

from cmd_tf.loss import flat_dice_coef_loss
from cmd_tf.model import get_xunet_model

exp_decay_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    decay_rate=0.9)

constant_decay_learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
constant_decay_learning_rate_boundaries = [125, 250, 500, 240000, 360000]
constant_decay_learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=constant_decay_learning_rate_boundaries, values=constant_decay_learning_rates
)

@dataclass
class RunConfig:
    name: str
    loss: any
    optimizer: any = "adam"
    model: any = None
    callbacks: list[tf.keras.callbacks.Callback] = []

# ---------------------------------------------------------------------------------------------------------------------------------------------

configs = [
    RunConfig( name="xunet-dice", loss=flat_dice_coef_loss, model=get_xunet_model((320, 320), 1)),
    RunConfig( name="bcross", loss="binary_crossentropy"),
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