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

from drivable.data import GetDrivableDataloader
from drivable.model import get_unet_model

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
# flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
# flags.DEFINE_bool("log_eval", False, "Log model prediction, needs --wandb argument as well.")

TRAIN_DATA_PATH = f"artifacts/bdd100k-dataset:v0/images/100k/train"
TRAIN_MASK_PATH = f"artifacts/train_masks:v0"
    
img_paths = glob.glob(f"{TRAIN_DATA_PATH}/*.jpg")
mask_paths = glob.glob(f"{TRAIN_MASK_PATH}/*.png")
print(img_paths[:5], mask_paths[:5])


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)

    CALLBACKS = []
    # # Initialize a Weights and Biases run.
    # if FLAGS.wandb:
    #     run = wandb.init(
    #         project=CONFIG.value.wandb_config.project,
    #         job_type='train',
    #         config=config.to_dict(),
    #     )
    #     wandb.run._label(code="ccd2022")
    #     # WandbCallback for experiment tracking
    #     CALLBACKS += [WandbCallback(save_model=False)]

    # Download and get dataset
    # dataset_name = config.dataset_config.dataset_name
    # info, (train_images, train_labels) = download_and_get_dataset(dataset_name, 'train')
    # info, (valid_images, valid_labels) = download_and_get_dataset(dataset_name, 'valid')


    # Get dataloader
    make_dataloader = GetDrivableDataloader(config)
    trainloader = make_dataloader.get_dataloader(img_paths, mask_paths)
    # validloader = make_dataloader.get_dataloader(valid_images, valid_labels, dataloader_type="valid")
    imgs, masks = next(iter(trainloader))
    print(imgs.shape, masks.shape)

    # # Get model
    tf.keras.backend.clear_session()
    model = get_unet_model((224, 224), 3)
    model.summary()

    # # Initialize callbacks
    # callback_config = config.callback_config
    # # Builtin early stopping callback
    # if callback_config.use_earlystopping:
    #     earlystopper = get_earlystopper(config)
    #     CALLBACKS += [earlystopper]
    # # Built in callback to reduce learning rate on plateau
    # if callback_config.use_reduce_lr_on_plateau:
    #     reduce_lr_on_plateau = get_reduce_lr_on_plateau(config)
    #     CALLBACKS += [reduce_lr_on_plateau]

    # # Initialize Custom W&B callbacks
    # if FLAGS.log_model:
    #     # Custom W&B model checkpoint callback
    #     model_checkpointer = get_model_checkpoint_callback(config)
    #     CALLBACKS += [model_checkpointer]

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
