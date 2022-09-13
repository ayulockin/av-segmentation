import os
import glob
import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

# Configs
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")

DRIVABLE_SEG_MAP = {0: "direct", 1: "alternative", 2: "background"}


def main(_):
    # Get configs for uploading the data
    config = CONFIG.value
    print(config)

    # Initialize W&B run
    run = wandb.init(
        entity="av-team",
        project="drivable-segmentation",
        job_type=f"{config.data_type}_table",
        config=config.to_dict(),
    )

    # Since the val split is from the train set
    true_data_type = "train"
    if config.data_type == "val":
        config.data_type = "train"
        true_data_type = "val"
    # Since the official val set is out test set
    if config.data_type == "test":
        config.data_type = "val"
        true_data_type = "test"

    # Paths to the images and masks
    DATA_PATH = f"artifacts/bdd100k-dataset:v0/images/100k/{config.data_type}"
    MASK_PATH = f"artifacts/{config.data_type}_masks:v0"

    # Get the split
    img_files = np.loadtxt(config.data_split_file, dtype=str)

    # Get the chunk based on the index:
    files_per_chunk = len(img_files) // config.chunks
    img_files = img_files[
        files_per_chunk * config.index : files_per_chunk * (config.index + 1)
    ]

    # Initialize the W&B Artifact with experimental incremental arg
    data_at = wandb.Artifact(name=true_data_type, type="dataset", incremental=True)

    # Use BDD100K dataset artifact for lineage
    bdd100k_artifact = run.use_artifact(
        "av-team/bdd100k-perception/bdd100k-dataset:v0", type="dataset"
    )
    # Use Mask artifact for lineage
    mask_artifact = run.use_artifact(
        f"av-team/drivable-segmentation/{config.data_type}_masks:v0", type="masks"
    )

    # Create a Partitioned Table pointing to a directory in the artifact (only
    # need to do this once)
    parts_dir = f"{true_data_type}_parts"
    if config.index == 0:
        table = wandb.data_types.PartitionedTable(parts_dir)
        data_at.add(table, true_data_type)

    # Intialize W&B Tables for drivable seg data
    column_names = ["image_id", "image"]
    data_table = wandb.Table(columns=column_names, allow_mixed_types=True)

    # Add data to the table row-wise
    for idx, file_name in tqdm(enumerate(img_files)):
        idx += files_per_chunk * config.index
        file_id = file_name.split(".")[0]
        if os.path.isfile(f"{MASK_PATH}/{file_id}.png"):
            # Get image
            image = Image.open(f"{DATA_PATH}/{file_name}")
            # Get mask
            mask = np.array(Image.open(f"{MASK_PATH}/{file_id}.png"))
            # Add image and mask to W&B table
            data_table.add_data(
                idx,
                wandb.Image(
                    image,
                    masks={
                        "ground_truth": {
                            "mask_data": mask,
                            "class_labels": DRIVABLE_SEG_MAP,
                        }
                    },
                ),
            )

    # Log the artifact to W&B
    table_path = f"{parts_dir}/table_{config.index}"
    data_at.add(data_table, table_path)
    data_at.add_file(config.data_split_file)
    run.log_artifact(data_at)


if __name__ == "__main__":
    app.run(main)
