import wandb
from tensorflow import keras
import glob
from ml_collections.config_flags import config_flags
from absl import flags
from absl import app

from drivable.data import GetDrivableDataloader
from drivable.model import get_unet_model
from drivable.utils.devices import initialize_device
from drivable import callbacks

api = wandb.Api()
artifact = api.artifact("av-team/drivable-segmentation/run_xl1w4kbt_model:v19")
artifact_dir = artifact.download()

model = keras.models.load_model(f"{artifact_dir}")

TEST_DATA_PATH = f"artifacts/bdd100k-dataset:v0/images/100k/val"
TEST_MASK_PATH = f"artifacts/val_masks:v0"

img_paths = glob.glob(f"{TEST_DATA_PATH}/*.jpg")
mask_paths = glob.glob(f"{TEST_MASK_PATH}/*.png")

FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")


def main(_):
    config = CONFIG.value
    make_dataloader = GetDrivableDataloader(config)
    print(len(img_paths), len(mask_paths))
    testloader = make_dataloader.get_dataloader(img_paths, mask_paths)
    model.evaluate(testloader)

if __name__ == "__main__":
    app.run(main)
