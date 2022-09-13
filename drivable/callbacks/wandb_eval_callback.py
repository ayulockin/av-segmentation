import wandb
import tensorflow as tf
from drivable.utils import BaseWandbEvalCallback


class WandbSegCallback(BaseWandbEvalCallback):
    def __init__(
        self,
        validation_data,
        id2label,
        num_samples=100,
    ):
        data_table_columns = ["idx", "ground_truth"]
        pred_table_columns = ["epoch"] + data_table_columns + ["prediction"]
        super().__init__(data_table_columns, pred_table_columns)

        # Make unbatched iterator from `tf.data.Dataset`.
        self.val_ds = validation_data.unbatch().take(num_samples)
        self.id2label = id2label

    def add_ground_truth(self, logs):
        for idx, (image, mask) in enumerate(self.val_ds.as_numpy_iterator()):
            self.data_table.add_data(
                idx,
                wandb.Image(
                    image,
                    masks={
                        "ground_truth": {
                            "mask_data": tf.squeeze(mask, axis=-1).numpy(),
                            "class_labels": self.id2label,
                        }
                    },
                ),
            )

    def add_model_predictions(self, epoch, logs):
        data_table_ref = self.data_table_ref
        table_idxs = data_table_ref.get_index()

        for idx, (image, mask) in enumerate(self.val_ds.as_numpy_iterator()):
            pred = self.model.predict(tf.expand_dims(image, axis=0), verbose=0)
            pred = tf.squeeze(tf.argmax(pred, axis=-1), axis=0)

            pred_wandb_mask = wandb.Image(
                image,
                masks={
                    "prediction": {"mask_data": pred.numpy(), "class_labels": self.id2label}
                },
            )
            self.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],
                data_table_ref.data[idx][1],
                pred_wandb_mask,
            )


def get_evaluation_callback(args, validation_data, id2label):
    return WandbSegCallback(
        validation_data, id2label, num_samples=args.callback_config.viz_num_images
    )
