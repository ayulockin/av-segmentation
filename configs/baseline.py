import os
import ml_collections
from ml_collections import config_dict


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.entity = "av-team"
    configs.project = "drivable-segmentation"

    return configs


def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.dataset_name = "BDD100K"
    configs.image_height = 224
    configs.image_width = 224
    configs.channels = 3
    configs.mask_channels = 1
    configs.apply_resize = True
    configs.batch_size = 128
    configs.num_classes = 3
    configs.apply_one_hot = True
    configs.do_cache = False
    configs.train_version = "latest"
    configs.val_version = "latest"
    configs.test_version = "latest"

    return configs


def get_model_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_name = "vanilla unet"
    configs.model_img_height = 224
    configs.model_img_width = 224
    configs.model_img_channels = 3
    configs.model_mask_channels = 1
    configs.dropout_rate = 0.5
    configs.post_gap_dropout = False

    return configs


def get_callback_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    # Early stopping
    configs.use_earlystopping = True
    configs.early_patience = 6
    # Reduce LR on plateau
    configs.use_reduce_lr_on_plateau = False
    configs.rlrp_factor = 0.2
    configs.rlrp_patience = 3
    # Model checkpointing
    configs.checkpoint_filepath = "wandb/model_{epoch}"
    configs.save_best_only = True
    # Model evaluation
    configs.viz_num_images = 100
    # Use tensorboard
    configs.use_tensorboard = True

    return configs


def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.epochs = 2
    configs.use_augmentations = False
    configs.optimizer = "adam"
    configs.learning_rate = 0.001
    configs.sgd_momentum = 0.9
    configs.loss = "sparse_categorical_crossentropy"

    return configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.model_config = get_model_configs()
    config.callback_config = get_callback_configs()
    config.train_config = get_train_configs()

    return config
