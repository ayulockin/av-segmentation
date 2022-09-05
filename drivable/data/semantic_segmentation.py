import os
import tensorflow as tf
from typing import Optional

from .base import SegmentationDatasetFactory


class BDDSemanticSegmentationLoader(SegmentationDatasetFactory):
    def __init__(
        self,
        image_size: int = 512,
        num_classes: Optional[int] = None,
        batch_size: int = 16,
        data_dir: Optional[str] = None,
    ):
        self.artifact_id = "av-team/bdd100k-perception/bdd100k-dataset:v0"
        self.data_dir = data_dir
        super().__init__(image_size, num_classes, batch_size)

    def _get_files(self, split: str):
        segmentation_masks = glob(
            os.path.join(self.data_dir, "labels/sem_seg/masks", split, "*")
        )
        images, masks = [], []
        for mask in segmentation_masks:
            image_file_100k = os.path.join(
                self.data_dir,
                "images/100k",
                split,
                os.path.basename(mask).replace("png", "jpg"),
            )
            image_file_10k = os.path.join(
                self.data_dir,
                "images/10k",
                split,
                os.path.basename(mask).replace("png", "jpg"),
            )
            if os.path.isfile(image_file_100k):
                images.append(image_file_100k)
                masks.append(mask)
            elif os.path.isfile(image_file_10k):
                images.append(image_file_10k)
                masks.append(mask)
        return images, masks

    def fetch_dataset(self):
        if self.data_dir is None:
            if wandb.run is None:
                api = wandb.Api()
                artifact = api.artifact(self.artifact_id, type="dataset")
            else:
                artifact = wandb.use_artifact(self.artifact_id, type="dataset")
            self.data_dir = artifact.download()
        self.train_image_files, self.train_mask_files = self._get_files(split="train")
        self.val_image_files, self.val_mask_files = self._get_files(split="val")

    def __len__(self):
        return len(self.train_image_files) + len(self.val_image_files)
