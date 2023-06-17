from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tensorflow import optimizers as optis
import keras
from keras import losses as loss
from keras.callbacks import LearningRateScheduler

from cmd_tf.model_xunet import flat_dice_coef_loss, get_xunet_model, get_xunet_traindata, get_xunet_valdata
from cmd_tf.models_sm import get_sm_unet_model, sm_dice_x_bfocal_loss, sm_metrics, sm_optim, get_sm_traindata, get_sm_valdata, get_sm_linknet_model, get_sm_fpn_model, get_sm_pspnet_model

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
    get_model: any
    dataset_loader: tuple
    metrics: any = None
    optimizer: any = "adam"
    callbacks: list[tf.keras.callbacks.Callback] = None

# ---------------------------------------------------------------------------------------------------------------------------------------------

configs = [
    RunConfig( 
              name="xunet", 
              get_model=get_xunet_model, 
              dataset_loader=(get_xunet_traindata, 
                              get_xunet_valdata), 
              loss=flat_dice_coef_loss,  ),
    RunConfig(
              name="sm-unet", 
              get_model=get_sm_unet_model, 
              dataset_loader=(get_sm_traindata,
                              get_sm_valdata),
              loss=sm_dice_x_bfocal_loss, 
              optimizer=sm_optim,
              metrics=sm_metrics,
              callbacks=[keras.callbacks.ReduceLROnPlateau(),]
              ),
    RunConfig(
              name="sm-linknet", 
              get_model=get_sm_linknet_model, 
              dataset_loader=(get_sm_traindata,
                              get_sm_valdata),
              loss=sm_dice_x_bfocal_loss,
              optimizer=sm_optim,
              metrics=sm_metrics,
              callbacks=[keras.callbacks.ReduceLROnPlateau(),]
              ),
    RunConfig(
              name="sm-fpn", 
              get_model=get_sm_fpn_model, 
              dataset_loader=(get_sm_traindata,
                              get_sm_valdata),
              loss=sm_dice_x_bfocal_loss, 
              optimizer=sm_optim,
              metrics=sm_metrics,
              callbacks=[keras.callbacks.ReduceLROnPlateau(),]
              ),
    RunConfig(
              name="sm-psnet", 
              get_model=get_sm_pspnet_model, 
              dataset_loader=(get_sm_traindata,
                              get_sm_valdata),
              loss=sm_dice_x_bfocal_loss, 
              optimizer=sm_optim,
              metrics=sm_metrics,
              callbacks=[keras.callbacks.ReduceLROnPlateau(),]
              ),
    ]

# ---------------------------------------------------------------------------------------------------------------------------------------------

def load_runconfig(runname: str = "", additional_settings = {}):
    cur_conf: RunConfig = None
    
    # Choose config
    configs.reverse()
    for conf in configs:
        if runname.startswith(conf.name):
            cur_conf = conf
            break
    if cur_conf is None:
        cur_conf = configs[-1] # First conf is default
        
    # Fill config none fields
    if cur_conf.callbacks is None:
        cur_conf.callbacks = []
    if cur_conf.metrics is None:
        cur_conf.metrics = [  tf.keras.metrics.BinaryAccuracy(), 
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.Precision(), ]
    cur_conf.additional_settings = additional_settings
    
    return cur_conf