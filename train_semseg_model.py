import os

import wandb.keras
from absl import app
from absl import flags
from absl import logging
from datetime import datetime

from ml_collections.config_flags import config_flags

from tensorflow.keras import losses, optimizers, callbacks
from tensorflow.keras import mixed_precision, applications

from drivable.data import BDDSemanticSegmentationLoader
from drivable.model import build_deeplabv3_plus
from drivable.utils.devices import initialize_device
from drivable.callbacks import WandBMetricsLogger, WandbModelCheckpoint


FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="wandb_project", default=None, help="Weights & Biases project name"
)
flags.DEFINE_string(
    name="wandb_entity", default=None, help="Weights & Biases entity name"
)
flags.DEFINE_string(
    name="wandb_job_type", default=None, help="Type of Weights & Biases job"
)
config_flags.DEFINE_config_file("experiment_configs")


def main(_):
    # Detect strategy
    strategy = utils.initialize_device()
    global_batch_size = (
        FLAGS.experiment_configs.data_configs.local_batch_size
        * strategy.num_replicas_in_sync
    )
    FLAGS.experiment_configs.data_configs["global_batch_size"] = global_batch_size

    # Set precision.
    if FLAGS.experiment_configs.training_configs.use_mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        logging.info("Using mixed-precision.")
        policy = mixed_precision.global_policy()
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"

    wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        job_type=FLAGS.wandb_job_type,
        config=FLAGS.experiment_configs.to_dict(),
    )

    data_loader = BDDSemanticSegmentationLoader(
        image_size=FLAGS.experiment_configs.data_configs.image_size,
        num_classes=FLAGS.experiment_configs.data_configs.num_classes
        if FLAGS.experiment_configs.training_configs.one_hot_encoding
        else None,
        batch_size=global_batch_size,
    )
    train_dataset, val_dataset = data_loader.get_datasets()

    backbone_type = None
    if FLAGS.experiment_configs.model_configs.backbone_alias == "resnet50":
        backbone_type = applications.ResNet50

    with strategy.scope():
        model = build_deeplabv3_plus(
            image_size=FLAGS.experiment_configs.data_configs.image_size,
            num_classes=FLAGS.experiment_configs.data_configs.num_classes,
            bakbone_type=backbone_type,
            encoder_backbone_layer=FLAGS.experiment_configs.model_configs.encoder_backbone_layer,
            decoder_backbone_layer=FLAGS.experiment_configs.model_configs.decoder_backbone_layer,
        )

        loss = None
        if FLAGS.experiment_configs.training_configs.loss_alias == "cross_entropy":
            loss = losses.SparseCategoricalCrossentropy(
                from_logits=not FLAGS.experiment_configs.training_configs.one_hot_encoding
            )

        optimizer = optimizers.Adam(
            learning_rate=FLAGS.experiment_configs.training_configs.learning_rate
        )

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    callbacks = [
        WandBMetricsLogger(),
        WandbModelCheckpoint(filepath="model", save_best_only=True),
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FLAGS.experiment_configs.training_configs.epochs,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    app.run(main)
