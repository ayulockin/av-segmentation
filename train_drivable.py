import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import wandb
from wandb.keras import WandbCallback
import tensorflow as tf

from drivable.data import download_dataset, preprocess_dataset
from drivable.data import GetDrivableDataloader
from drivable.model import get_unet_model
from drivable.utils.devices import initialize_device
from drivable import callbacks

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
# flags.DEFINE_bool("log_eval", False, "Log model prediction, needs --wandb argument as well.")


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)

    # Detect strategy
    strategy = initialize_device()
    batch_size = (
        config.dataset_config.batch_size
        * strategy.num_replicas_in_sync
    )
    config.dataset_config.batch_size = batch_size

    CALLBACKS = []
    # Initialize a Weights and Biases run.
    if FLAGS.wandb:
        run = wandb.init(
            entity=CONFIG.value.wandb_config.entity,
            project=CONFIG.value.wandb_config.project,
            job_type='train',
            config=config.to_dict(),
        )
        # Initialize W&B metrics logger callback.
        CALLBACKS += [callbacks.WandBMetricsLogger()]

    # Download and get dataset
    # dataset_name = config.dataset_config.dataset_name
    train_df = download_and_get_dataset(dataset_name, 'train')
    valid_df = download_and_get_dataset(dataset_name, 'valid')

    train_imgs, train_masks = preprocess_dataset(train_df)
    valid_imgs, valid_masks = preprocess_dataset(valid_df)

    # Get dataloader
    make_dataloader = GetDrivableDataloader(config)
    trainloader = make_dataloader.get_dataloader(train_imgs, train_masks)
    validloader = make_dataloader.get_dataloader(valid_imgs, valid_masks, dataloader_type="valid")

    with strategy.scope():
        # Get model
        tf.keras.backend.clear_session()
        model = get_unet_model((224, 224), 3)

        # if config.train_config.loss == "sparse_categorical_crossentropy":
        #     loss = tf.keras.losses.SparseCategoricalCrossentropy()

        # if config.train_config.optimizer == "adam":
        #     optimizer = tf.keras.optimizers.Adam(
                
        #     )

    # Initialize callbacks
    callback_config = config.callback_config
    # Builtin early stopping callback
    if callback_config.use_earlystopping:
        earlystopper = callbacks.get_earlystopper(config)
        CALLBACKS += [earlystopper]
    # Built in callback to reduce learning rate on plateau
    if callback_config.use_reduce_lr_on_plateau:
        reduce_lr_on_plateau = callbacks.get_reduce_lr_on_plateau(config)
        CALLBACKS += [reduce_lr_on_plateau]
    
    # Initialize Model checkpointing callback
    if FLAGS.log_model:
        # Custom W&B model checkpoint callback
        model_checkpointer = callbacks.get_model_checkpoint_callback(config)
        CALLBACKS += [model_checkpointer]    

    # # Custom W&B model prediction visualization callback
    # if wandb.run is not None:
    #     if FLAGS.log_eval:
    #         model_pred_viz = get_evaluation_callback(config, validloader)
    #         CALLBACKS += [model_pred_viz]

    # Compile the model
    model.compile(
        optimizer = config.train_config.optimizer,
        loss = config.train_config.loss,
        metrics = config.train_config.metrics
    )

    # Train the model
    model.fit(
        trainloader,
        # validation_data = validloader,
        epochs = config.train_config.epochs,
        callbacks=CALLBACKS
    )


if __name__ == "__main__":
    app.run(main)
