import ml_collections


def get_data_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.image_size = 512
    config.num_classes = 20
    config.local_batch_size = 16

    return config


def get_model_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.backbone_alias = "resnet50"
    config.encoder_backbone_layer = "conv4_block6_2_relu"
    config.decoder_backbone_layer = "conv2_block3_2_relu"

    return config


def get_training_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.loss_alias = "cross_entropy"
    config.one_hot_encoding = False
    config.learning_rate = 1e-3
    config.epochs = 50

    return config


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.data_configs = get_data_configs()
    config.model_configs = get_model_configs()
    config.training_configs = get_training_configs()

    return config
