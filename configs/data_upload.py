import os
import ml_collections
from ml_collections import config_dict


def get_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.entity = "av-team"
    configs.project = "drivable-segmentation"
    configs.chunks = 10
    configs.index = 9 # 0 - 9 -> chunk 1st -> chunk 10th
    configs.data_type = "train" # "val"
    configs.data_split_file = "splits/train_split.txt"
    
    return configs
