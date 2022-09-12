import wandb


run = wandb.init(
    entity="av-team", project="drivable-segmentation", job_type="download_data"
)

artifact = run.use_artifact(
    "av-team/bdd100k-perception/bdd100k-dataset:v0", type="dataset"
)
artifact_dir = artifact.download()
