import wandb
from drivable.callbacks import BaseWandbEvalCallback


class WandbEvalCallback(BaseWandbEvalCallback):
    def __init__(
        self,
        validation_data,
        data_table_columns,
        pred_table_columns,
        id2label,
        num_samples=100,
    ):
        super().__init__(data_table_columns, pred_table_columns)

        # Make unbatched iterator from `tf.data.Dataset`.
        self.val_ds = validation_data.unbatch().take(num_samples)

        self.id2label = id2label

    def add_ground_truth(self):
        for idx, (image, mask) in enumerate(self.val_ds.as_numpy_iterator()):
            self.data_table.add_data(
                idx,
                wandb.Image(
                    image,
                    masks={
                        "ground_truth": {
                            "mask_data": mask,
                            "class_labels": self.id2label,
                        }
                    },
                ),
            )

    def add_model_predictions(self, epoch):
        data_table_ref = self.data_table_ref
        table_idxs = data_table_ref.get_index()

        for idx, (image, mask) in enumerate(self.val_ds.as_numpy_iterator()):
            pred = self.model.predict(tf.expand_dims(image, axis=0))

            pred_wandb_mask = wandb.Image(
                image,
                masks={
                    "prediction": {"mask_data": pred, "class_labels": self.id2label}
                },
            )
            self.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],
                data_table_ref.data[idx][1],
                pred_wandb_mask,
            )
