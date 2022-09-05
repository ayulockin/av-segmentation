import tensorflow as tf
from functools import partial
from typing import List, Optional
from abc import ABC, abstractmethod


class SegmentationDatasetFactory(ABC):
    def __init__(
        self,
        image_size: int = 512,
        num_classes: Optional[int] = None,
        batch_size: int = 16,
    ):
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.fetch_dataset()
        assert len(self.train_image_files) == len(self.train_mask_files)
        assert len(self.val_image_files) == len(self.val_mask_files)

    @abstractmethod
    def fetch_dataset(self):
        pass

    def random_crop(self, image, label):
        image_shape = tf.shape(image)[:2]
        crop_width = tf.random.uniform(
            shape=(), maxval=image_shape[1] - self.image_size + 1, dtype=tf.int32
        )
        crop_height = tf.random.uniform(
            shape=(), maxval=image_shape[0] - self.image_size + 1, dtype=tf.int32
        )
        image_cropped = image[
            crop_height : crop_height + self.image_size,
            crop_width : crop_width + self.image_size,
        ]
        label_cropped = label[
            crop_height : crop_height + self.image_size,
            crop_width : crop_width + self.image_size,
        ]
        return image_cropped, label_cropped

    def read_image(self, image_path, is_label: bool):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1 if is_label else 3)
        image.set_shape([None, None, 1 if is_label else 3])
        image = tf.cast(image, dtype=tf.float32)
        image = image / 127.5 - 1 if not is_label else image
        return image

    def load_data(self, image, label):
        image = self.read_image(image, is_label=False)
        label = self.read_image(label, is_label=True)
        image = tf.image.resize(images=image, size=[self.image_size, self.image_size])
        label = tf.image.resize(images=label, size=[self.image_size, self.image_size])
        # label = (
        #     tf.one_hot(label, depth=self.num_classes, axis=-1)
        #     if self.num_classes is not None
        #     else label
        # )
        return image, label

    def build_dataset(self, image_list: List[str], label_list: List[str]):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        dataset = dataset.map(
            map_func=self.load_data, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset.prefetch(tf.data.AUTOTUNE)
