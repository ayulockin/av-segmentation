import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import wandb
from wandb.keras import WandbCallback
import tensorflow as tf

from drivable.data import download_dataset, preprocess_dataframe
from drivable.data import GetDrivableDataloader
from drivable.model import get_unet_model
from drivable.utils.devices import initialize_device
from drivable import callbacks

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
flags.DEFINE_bool(
    "log_eval", False, "Log model prediction, needs --wandb argument as well."
)

# ID to Label dict
DRIVABLE_SEG_MAP = {0: "direct", 1: "alternative", 2: "background"}


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)

    # Detect strategy
    strategy = initialize_device()
    batch_size = config.dataset_config.batch_size * strategy.num_replicas_in_sync
    config.dataset_config.batch_size = batch_size

    CALLBACKS = []
    # Initialize a Weights and Biases run.
    if FLAGS.wandb:
        run = wandb.init(
            entity=CONFIG.value.wandb_config.entity,
            project=CONFIG.value.wandb_config.project,
            job_type="train",
            config=config.to_dict(),
        )
        # Initialize W&B metrics logger callback.
        CALLBACKS += [callbacks.WandBMetricsLogger()]

    # Download and get dataset
    train_df = download_dataset("train", version=config.dataset_config.train_version)
    valid_df = download_dataset("val", version=config.dataset_config.val_version)

    train_imgs, train_masks = preprocess_dataframe(train_df)
    valid_imgs, valid_masks = preprocess_dataframe(valid_df)
    print("size of data: ", len(train_imgs), len(valid_imgs))

    # Get dataloader
    make_dataloader = GetDrivableDataloader(config)
    trainloader = make_dataloader.get_dataloader(train_imgs, train_masks)
    validloader = make_dataloader.get_dataloader(
        valid_imgs, valid_masks, dataloader_type="valid"
    )

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

    # Custom W&B model prediction visualization callback
    if wandb.run is not None:
        if FLAGS.log_eval:
            model_pred_viz = callbacks.get_evaluation_callback(
                config, validloader, DRIVABLE_SEG_MAP
            )
            CALLBACKS += [model_pred_viz]

    with strategy.scope():
        # Get model
        model = get_unet_model(
            (config.model_config.model_img_height, config.model_config.model_img_width),
            config.model_config.model_img_channels,
        )

        # Optimizer
        if config.train_config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.train_config.learning_rate
            )

        # Loss function
        if config.train_config.loss == "sparse_categorical_crossentropy":
            loss = tf.keras.losses.SparseCategoricalCrossentropy()

        # Metrics
        metrics = []
        metrics += [
            tf.keras.metrics.OneHotIoU(
                num_classes=config.dataset_config.num_classes,
                target_class_ids=[0],
            ),
            tf.keras.metrics.OneHotMeanIoU(num_classes=config.dataset_config.num_classes),
            "accuracy",
        ]
        print(metrics)

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    model.fit(
        trainloader,
        validation_data=validloader,
        epochs=config.train_config.epochs,
        callbacks=CALLBACKS,
    )


if __name__ == "__main__":
    app.run(main)
