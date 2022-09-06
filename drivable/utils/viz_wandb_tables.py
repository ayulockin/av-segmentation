import os
import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image

DATA_TYPE = "train"
DATA_SIZE = 30000

DATA_PATH = f"artifacts/bdd100k-dataset:v0/images/100k/{DATA_TYPE}"
MASK_PATH = f"artifacts/{DATA_TYPE}_masks:v0"
print(os.path.isdir(DATA_PATH), os.path.isdir(MASK_PATH))

img_files = os.listdir(DATA_PATH)
# Random sample of images
img_files = np.random.choice(
    img_files,
    size=DATA_SIZE,
    replace=False
)
# Save the split as txt file
np.savetxt(
    f"{DATA_TYPE}_split.txt",
    img_files,
    fmt="%s",
    delimiter=","
)

DRIVABLE_SEG_MAP = {
    0: "direct",
    1: "alternative",
    2: "background"
}

# Initialize W&B run
run = wandb.init(
    entity="av-team",
    project="drivable-segmentation",
    job_type=f"{DATA_TYPE}_table"
)

data_at = wandb.Artifact(name=DATA_TYPE, type="dataset")
# Use BDD100K dataset artifact
bdd100k_artifact = run.use_artifact('av-team/bdd100k-perception/bdd100k-dataset:v0', type='dataset')
# Use Mask artifact
mask_artifact = run.use_artifact(f'av-team/drivable-segmentation/{DATA_TYPE}_masks:v0', type='masks')

# Intialize W&B Tables for drivable seg data
column_names = [
    "image_id",
    "image"
]
data_table = wandb.Table(columns=column_names, allow_mixed_types=True)

# Add data to the table row-wise
masks_unfound = []
for idx, file_name in tqdm(enumerate(img_files)):
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
                masks = {
                    "ground_truth": {
                        "mask_data": mask,
                        "class_labels": DRIVABLE_SEG_MAP
                    }
                }
            )
        )
    else:
        masks_unfound.append(file_id)

# Log the artifact to W&B
data_at.add(data_table, name = f"{DATA_TYPE}_table")
data_at.add_file(f"{DATA_TYPE}_split.txt")
run.log_artifact(data_at)
