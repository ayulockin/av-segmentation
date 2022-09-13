import wandb
import os
import glob
import json
import pandas as pd
from tqdm import tqdm


def download_dataset(dataset_name: str,
                     dataset_type: str = 'dataset',
                     version: str='latest',
                     save_at='artifacts/'):
    """
    Utility function to download the data saved as W&B artifacts and return a dataframe
    with path to the dataset and associated label.

    Args:
        dataset_name (str): The name of the dataset - `train`, `val`, `test`.
        dataset_type (str): The type of the dataset -  `dataset`.
        version (str): The version of the dataset to be downloaded. By default it's `latest`,
            but you can provide different version as `vX`, where, X can be 0,1,...
            
        Note that the following combination of dataset_name and dataset_type are valid:
            - `train`, `dataset`
            - `val`, `dataset`
            - `test`, `dataset`

        # TODO (ayulockin): Add GTA5 downloading script

    Return:
        df_data (pandas.DataFrame): Dataframe with path to images and masks
    """
    if dataset_name == 'train' and os.path.exists(save_at+'train.csv'):
        data_df = pd.read_csv(save_at+'train.csv')
    elif dataset_name == 'val' and os.path.exists(save_at+'valid.csv'):
        data_df = pd.read_csv(save_at+'valid.csv')
    elif dataset_name == 'test' and os.path.exists(save_at+'test.csv'):
        data_df = pd.read_csv(save_at+'test.csv')
    else:
        data_df = None
        print('Downloading dataset...')

    if data_df is None:
        # Download the dataset.
        wandb_api = wandb.Api()
        artifact = wandb_api.artifact(f'av-team/drivable-segmentation/{dataset_name}:{version}', type=dataset_type)
        artifact_dir = artifact.download()

        # Open the W&B table downloaded as a json file.
        partition_file = glob.glob(artifact_dir+'/*.json')
        assert len(partition_file) == 1
        with open(partition_file[0]) as f:
            partition_data = json.loads(f.read())
            assert partition_data['_type'] == "partitioned-table"
            # assert partition_data['parts_path'] == f"{dataset_type}_parts"

        # Get parts table
        part_paths = partition_data['parts_path']
        part_tables = glob.glob(artifact_dir+f"/{part_paths}/*.json")

        data = []
        for parts in part_tables:
            with open(parts) as f:
                part_data = json.loads(f.read())
                columns = part_data["columns"]
                data.extend(part_data["data"])

        # Create a dataframe with path and label
        df_columns = ['image_id', 'image_path', 'mask_path', 'width', 'height']
        data_df = pd.DataFrame(columns=df_columns)

        for idx, example in tqdm(enumerate(data)):
            image_id = int(example[0])
            image_dict = example[1]
            height = image_dict.get('height')
            width = image_dict.get('width')
            image_path = os.path.join(artifact_dir, image_dict["path"])
            mask_path = os.path.join(artifact_dir, image_dict["masks"]["ground_truth"]["path"])
            
            df_data = [image_id, image_path, mask_path, width, height]
            data_df.loc[idx] = df_data
            
    # Shuffle the dataframe
    if dataset_name == 'train':
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the dataframes as csv
    if dataset_name == 'train' and not os.path.exists(save_at+'train.csv'):
        data_df.to_csv(save_at+'train.csv', index=False)

    if dataset_name == 'val' and not os.path.exists(save_at+'valid.csv'):
        data_df.to_csv(save_at+'valid.csv', index=False)

    if dataset_name == 'test' and not os.path.exists(save_at+'test.csv'):
        data_df.to_csv(save_at+'test.csv', index=False)

    return data_df


def preprocess_dataframe(df):
    # Remove unnecessary columns
    df = df.drop(['image_id', 'width', 'height'], axis=1)
    assert len(df.columns) == 2

    image_paths = df.image_path.values
    mask_paths = df.mask_path.values

    return image_paths, mask_paths
