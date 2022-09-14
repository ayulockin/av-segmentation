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
from drivable.model import get_unet_model, download_model
from drivable.utils.devices import initialize_device
from drivable import callbacks

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_string(
    "model_artifact_path", None, "Model checkpoint saved as W&B artifact."
)
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
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
            job_type="evaluate",
            config=config.to_dict(),
        )
        # Initialize W&B metrics logger callback.
        CALLBACKS += [callbacks.WandBMetricsLogger()]

    # Download and get dataset
    test_df = download_dataset("test", version=config.dataset_config.test_version)
    test_imgs, test_masks = preprocess_dataframe(test_df)

    # Get dataloader
    make_dataloader = GetDrivableDataloader(config)
    testloader = make_dataloader.get_dataloader(
        test_imgs, test_masks, dataloader_type="test"
    )

    # Custom W&B model prediction visualization callback
    if wandb.run is not None:
        if FLAGS.log_eval:
            model_pred_viz = callbacks.get_evaluation_callback(
                config, testloader, DRIVABLE_SEG_MAP, is_train=False
            )
            CALLBACKS += [model_pred_viz]

    if FLAGS.model_artifact_path is not None:
        model_path = download_model(FLAGS.model_artifact_path)
        if wandb.run is not None:
            artifact = run.use_artifact(FLAGS.model_artifact_path, type='model')
        print("Path to the model checkpoint: ", model_path)
    else:
        raise

    with strategy.scope():
        # Get model
        model = tf.keras.models.load_model(model_path)

    # Train the model
    metrics = model.evaluate(testloader, callbacks=CALLBACKS, return_dict=True)
    print(metrics)

    if wandb.run is not None:
        wandb.log(metrics)


if __name__ == "__main__":
    app.run(main)
